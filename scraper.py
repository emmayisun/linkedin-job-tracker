#!/usr/bin/env python3
"""
LinkedIn Job Tracker — Scrapes PM job listings and generates enterprise AI fit analysis.
Designed to run as a GitHub Action on an hourly cron schedule.
"""

import csv
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from playwright.sync_api import sync_playwright

# --- Configuration ---
LINKEDIN_SEARCH_URL = (
    "https://www.linkedin.com/jobs/search/"
    "?alertAction=viewjobs"
    "&distance=25"
    "&f_TPR=r3600"
    "&geoId=90000084"
    "&keywords=product%20manager"
    "&origin=JOB_SEARCH_PAGE_JOB_FILTER"
    "&sortBy=DD"
    "&spellCorrectionEnabled=true"
)

CSV_FILE = Path(__file__).parent / "jobs.csv"
EMAIL_HTML_FILE = Path(__file__).parent / "email_body.html"

CSV_HEADERS = [
    "timestamp",
    "job_id",
    "job_title",
    "company",
    "location",
    "experience_years",
    "salary",
    "comment",
    "job_url",
]

GEMINI_PROMPT_TEMPLATE = """You are evaluating a job posting for someone who wants to do enterprise AI product management.

Job Title: {title}
Company: {company}
Job Description (excerpt):
{description}

Briefly evaluate:
1. What does this company do and how reputable is it?
2. Does this role involve enterprise AI product management?
3. Rate the fit as **High**, **Medium**, or **Low** for someone targeting enterprise AI PM roles.

Start your response with the rating on its own line, like:
Rating: High

Then provide a 2-3 sentence explanation. Keep your total response under 80 words."""


def load_existing_job_ids() -> set:
    """Load job IDs already tracked in the CSV to avoid duplicates."""
    if not CSV_FILE.exists():
        return set()
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["job_id"] for row in reader if "job_id" in row}


def parse_experience(text: str) -> str:
    """Extract years-of-experience requirements from job description text."""
    patterns = [
        r"(\d+)\+?\s*[-–]?\s*(\d+)?\+?\s*years?\s+(?:of\s+)?(?:relevant\s+)?(?:professional\s+)?experience",
        r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:relevant\s+)?(?:professional\s+)?experience",
        r"(\d+)\s*[-–]\s*(\d+)\s*years?",
        r"(\d+)\+\s*years?",
        r"minimum\s+(?:of\s+)?(\d+)\s*years?",
        r"at\s+least\s+(\d+)\s*years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = [g for g in match.groups() if g is not None]
            if len(groups) == 2:
                return f"{groups[0]}-{groups[1]} years"
            return f"{groups[0]}+ years"
    return "Not specified"


def parse_salary(card_meta: list[str], description: str) -> str:
    """Extract salary information from card metadata or job description."""
    # First check card metadata (more structured)
    salary_pattern = r"\$[\d,]+\.?\d*[Kk]?(?:/yr)?(?:\s*[-–to]+\s*\$[\d,]+\.?\d*[Kk]?(?:/yr)?)?"
    for meta in card_meta:
        match = re.search(salary_pattern, meta)
        if match:
            return match.group(0)

    # Fallback: check description
    match = re.search(salary_pattern, description)
    if match:
        return match.group(0)

    return "Not listed"


def scrape_jobs(li_at_cookie: str) -> list[dict]:
    """Use Playwright to scrape LinkedIn job search results."""
    jobs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        )

        # Set LinkedIn auth cookie
        context.add_cookies(
            [
                {
                    "name": "li_at",
                    "value": li_at_cookie,
                    "domain": ".linkedin.com",
                    "path": "/",
                }
            ]
        )

        page = context.new_page()
        print(f"Navigating to LinkedIn job search...")
        page.goto(LINKEDIN_SEARCH_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)  # Wait for JS rendering

        # Check if we're logged in by looking for job cards
        job_cards = page.query_selector_all("[data-job-id]")
        if not job_cards:
            print("WARNING: No job cards found. Cookie may be expired or page layout changed.")
            # Take a debug screenshot
            page.screenshot(path="debug_screenshot.png")
            browser.close()
            return []

        print(f"Found {len(job_cards)} job cards")

        for i, card in enumerate(job_cards):
            job_id = card.get_attribute("data-job-id")
            if not job_id:
                continue

            # Extract basic info from card
            title_el = card.query_selector('a[href*="/jobs/view/"]')
            company_el = card.query_selector(".artdeco-entity-lockup__subtitle")
            meta_items = card.query_selector_all("li")

            # Extract title from the <strong> tag inside the link to avoid
            # duplicated text (LinkedIn renders title in <strong> + a hidden
            # <span class="visually-hidden"> with "with verification" suffix)
            if title_el:
                strong_el = title_el.query_selector("strong")
                title = strong_el.inner_text().strip() if strong_el else title_el.inner_text().strip()
            else:
                title = "Unknown"

            company = company_el.inner_text().strip() if company_el else "Unknown"
            card_meta = [li.inner_text().strip() for li in meta_items]

            # Extract location from first meta item
            location = card_meta[0] if card_meta else "Unknown"

            # Build job URL
            job_url = f"https://www.linkedin.com/jobs/view/{job_id}/"

            # Click the card to load full job description
            print(f"  [{i+1}/{len(job_cards)}] Loading: {title} @ {company}")
            try:
                card.click()
                page.wait_for_timeout(random.randint(2000, 4000))

                # Wait for description panel to load
                page.wait_for_selector("#job-details", timeout=10000)

                # Extract full description
                desc_el = page.query_selector("#job-details")
                description = desc_el.inner_text().strip() if desc_el else ""

                # Extract salary from top card badges
                salary_from_detail = ""
                badge_els = page.query_selector_all(
                    ".job-details-jobs-unified-top-card__job-insight-view-model-secondary span"
                )
                for badge in badge_els:
                    text = badge.inner_text().strip()
                    if "$" in text and "/yr" in text.lower():
                        salary_from_detail = text
                        break

            except Exception as e:
                print(f"    Error loading description: {e}")
                description = ""
                salary_from_detail = ""

            # Parse salary (card metadata first, then detail panel, then description)
            salary_meta = card_meta + ([salary_from_detail] if salary_from_detail else [])
            salary = parse_salary(salary_meta, description)

            # Parse experience
            experience = parse_experience(description)

            jobs.append(
                {
                    "job_id": job_id,
                    "job_title": title,
                    "company": company,
                    "location": location,
                    "experience_years": experience,
                    "salary": salary,
                    "description": description[:3000],  # Truncate for Gemini
                    "job_url": job_url,
                }
            )

        browser.close()

    return jobs


def generate_comments(jobs: list[dict], api_key: str) -> list[dict]:
    """Use Gemini 2.5 Flash to generate enterprise AI fit comments."""
    client = genai.Client(api_key=api_key)

    for job in jobs:
        prompt = GEMINI_PROMPT_TEMPLATE.format(
            title=job["job_title"],
            company=job["company"],
            description=job["description"],
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt,
            )
            job["comment"] = response.text.strip()
        except Exception as e:
            print(f"  Gemini error for {job['job_title']}: {e}")
            job["comment"] = "Error generating comment"

        # Rate limit: small delay between API calls
        time.sleep(1)

    return jobs


def extract_rating(comment: str) -> str:
    """Extract High/Medium/Low rating from Gemini comment."""
    match = re.search(r"Rating:\s*(High|Medium|Low)", comment, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    # Fallback: look for the words anywhere
    for rating in ["High", "Medium", "Low"]:
        if rating.lower() in comment.lower():
            return rating
    return "Unknown"


def save_to_csv(jobs: list[dict]):
    """Append new jobs to the CSV file."""
    file_exists = CSV_FILE.exists()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "job_id": job["job_id"],
                    "job_title": job["job_title"],
                    "company": job["company"],
                    "location": job["location"],
                    "experience_years": job["experience_years"],
                    "salary": job["salary"],
                    "comment": job["comment"],
                    "job_url": job["job_url"],
                }
            )
    print(f"Saved {len(jobs)} jobs to {CSV_FILE}")


def generate_email_html(jobs: list[dict]) -> str:
    """Generate a styled HTML email with the job results table."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rating_colors = {
        "High": "#d4edda",    # green
        "Medium": "#fff3cd",  # yellow
        "Low": "#f8d7da",     # red
        "Unknown": "#e2e3e5", # gray
    }

    rows_html = ""
    for job in jobs:
        rating = extract_rating(job["comment"])
        bg_color = rating_colors.get(rating, "#e2e3e5")
        # Remove the "Rating: X" line from display comment
        display_comment = re.sub(r"Rating:\s*(High|Medium|Low)\s*\n?", "", job["comment"], flags=re.IGNORECASE).strip()

        rows_html += f"""
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">
                <a href="{job['job_url']}" style="color: #0073b1; text-decoration: none; font-weight: bold;">
                    {job['job_title']}
                </a>
            </td>
            <td style="padding: 10px; border: 1px solid #ddd;">{job['company']}</td>
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{job['experience_years']}</td>
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{job['salary']}</td>
            <td style="padding: 10px; border: 1px solid #ddd; background-color: {bg_color};">
                <strong>[{rating}]</strong> {display_comment}
            </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;">

<h2 style="color: #333;">LinkedIn PM Job Alert</h2>
<p style="color: #666;">Scan time: {timestamp} | Region: San Francisco Bay Area | Filter: Past 1 hour</p>
<p style="color: #666;">Found <strong>{len(jobs)}</strong> new job(s)</p>

{f'''<table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <thead>
        <tr style="background-color: #0073b1; color: white;">
            <th style="padding: 12px; border: 1px solid #ddd; text-align: left; width: 20%;">Job Title</th>
            <th style="padding: 12px; border: 1px solid #ddd; text-align: left; width: 12%;">Company</th>
            <th style="padding: 12px; border: 1px solid #ddd; text-align: center; width: 10%;">Experience</th>
            <th style="padding: 12px; border: 1px solid #ddd; text-align: center; width: 13%;">Salary</th>
            <th style="padding: 12px; border: 1px solid #ddd; text-align: left; width: 45%;">Comment (Enterprise AI Fit)</th>
        </tr>
    </thead>
    <tbody>
        {rows_html}
    </tbody>
</table>''' if jobs else '<p style="color: #999; font-style: italic;">No new jobs found in the past hour.</p>'}

<hr style="margin-top: 30px; border: none; border-top: 1px solid #eee;">
<p style="font-size: 12px; color: #999;">
    Generated by <a href="https://github.com/emmayisun/linkedin-job-tracker">linkedin-job-tracker</a> GitHub Action
</p>

</body>
</html>"""

    return html


def main():
    # Load secrets from environment
    li_at_cookie = os.environ.get("LI_AT_COOKIE", "")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    if not li_at_cookie:
        print("ERROR: LI_AT_COOKIE environment variable not set")
        return
    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return

    # Load existing job IDs for deduplication
    existing_ids = load_existing_job_ids()
    print(f"Loaded {len(existing_ids)} existing job IDs from CSV")

    # Scrape LinkedIn
    print("\n--- Scraping LinkedIn ---")
    all_jobs = scrape_jobs(li_at_cookie)
    print(f"Scraped {len(all_jobs)} total jobs")

    # Filter out duplicates
    new_jobs = [j for j in all_jobs if j["job_id"] not in existing_ids]
    print(f"After deduplication: {len(new_jobs)} new jobs")

    if not new_jobs:
        print("No new jobs found this run.")
        email_html = generate_email_html([])
        EMAIL_HTML_FILE.write_text(email_html, encoding="utf-8")
        Path(Path(__file__).parent / "has_new_jobs.txt").write_text("false")
        print("Done — nothing to report.")
        return

    # Generate Gemini comments
    print("\n--- Generating Gemini comments ---")
    new_jobs = generate_comments(new_jobs, gemini_api_key)

    # Save to CSV
    print("\n--- Saving to CSV ---")
    save_to_csv(new_jobs)

    # Generate email HTML
    print("\n--- Generating email ---")
    email_html = generate_email_html(new_jobs)
    EMAIL_HTML_FILE.write_text(email_html, encoding="utf-8")
    print(f"Email HTML written to {EMAIL_HTML_FILE}")

    # Write flag for workflow
    Path(Path(__file__).parent / "has_new_jobs.txt").write_text("true")

    # Summary
    print("\n--- Summary ---")
    for job in new_jobs:
        rating = extract_rating(job["comment"])
        print(f"  [{rating}] {job['job_title']} @ {job['company']} | {job['salary']}")


if __name__ == "__main__":
    main()
