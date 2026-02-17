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
def get_search_url() -> str:
    """Build LinkedIn search URL with appropriate time filter.

    PST 6am (UTC 14:00) run uses 9-hour lookback (r32400) to catch overnight jobs.
    All other runs use 1-hour lookback (r3600).
    """
    run_hour_utc = os.environ.get("RUN_HOUR_UTC", "")
    if run_hour_utc == "14":
        # Morning catch-up: 9 hours = 32400 seconds
        time_filter = "r32400"
        print("Morning catch-up mode: looking back 9 hours")
    else:
        # Normal hourly run: 1 hour = 3600 seconds
        time_filter = "r3600"
        print("Hourly mode: looking back 1 hour")

    return (
        "https://www.linkedin.com/jobs/search/"
        "?alertAction=viewjobs"
        "&distance=25"
        f"&f_TPR={time_filter}"
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

CANDIDATE_RESUME = """
CANDIDATE: Yi "Emma" Sun | San Francisco, CA

WORK EXPERIENCE:
- Commure (Enterprise AI, Series D Unicorn) — Product Manager, AI-Powered Enterprise Platforms (Oct 2023–Apr 2024)
  * Fine-tuned LLMs on 3 years of historical claims data to auto-generate CPT/ICD codes with medical necessity evidence, human-in-the-loop review workflow. Scaled to 150+ healthcare orgs, 20% revenue increase, 25% reduction in denials.
  * ML-driven customer health scoring, closed $2M revenue gaps, reduced churn 50%.
- JPMorgan Securities — PM & AI Strategy, Research Portal (Sep 2021–Sep 2023)
  * Built end-to-end audio-to-signal AI pipeline, improved LLM fine-tuning precision 30%, drove $5M revenue increase.
  * Led data strategy for trade-recommendation engine, increased trading volume 30% and profit 45%.
- Société Générale — PM, Trading Platform & Algorithms (Feb 2019–Jul 2021)
  * Spearheaded Single-Dealer Platform, 300% active user increase, 170% trading flow growth, 120% profitability uplift.

AI PRODUCT EXPERIENCE:
- ChatGeneT (May–Nov 2025): AI "Patient Simulator" agent deployed across 30+ hospitals. Fine-tuned model: 0.31% hallucination rate, 0.87 anthropomorphism score. CSAT 4.5/5, served 500+ junior doctors.
- Decoding the Beige Book (ACM ICAIF '25): End-to-end "text-to-signal" LLM pipeline, multi-model approach (GPT/Claude/Gemini/FinBERT/Mistral), peak F1 score 0.89 in recession nowcasting.

EDUCATION:
- Georgia Tech, MS in Computer Science, AI concentration (GPA 4.0) — Sep 2024–Dec 2025
- Columbia University, MA in Mathematics of Finance (GPA 3.84) — Sep 2017–Dec 2018

SKILLS:
- AI/ML: PyTorch, LangChain, LangGraph, LLM fine-tuning, prompt engineering, LLM evaluation
- Product: Figma, user research, PRDs, prioritization/roadmapping, A/B Testing, Agile/Scrum, Jira/Linear
- Analytics: Python, SQL, Looker, Tableau, Power BI, Retool
"""

GEMINI_BATCH_PROMPT_TEMPLATE = """You are a career advisor. Match the candidate's resume against each job posting below.

=== CANDIDATE RESUME ===
{resume}

=== JOB POSTINGS ===
{jobs_block}

=== INSTRUCTIONS ===
For each job, evaluate whether the candidate's skills and experience meet the job's stated requirements.
Focus ONLY on:
- Does the candidate have the specific technical skills the job asks for?
- Does the candidate have relevant domain experience the job requires?
- Are there critical required skills/experiences the candidate is missing?

Do NOT consider: whether the candidate is a student, work tenure length, or career gaps.

Return your response as a valid JSON array. Each element must have exactly these fields:
- "job_id": the job ID string
- "rating": exactly one of "High", "Medium", or "Low"
- "bullets": an array of exactly 3 short strings (each under 20 words)

The 3 bullets should cover:
1. Key matching skill/experience
2. Another matching or relevant qualification
3. Main gap or missing requirement (or "No major gaps" if strong match)

Example format:
[
  {{"job_id": "123", "rating": "High", "bullets": ["Has direct LLM fine-tuning experience matching the role", "Enterprise AI PM at Commure aligns with B2B focus", "No major gaps"]}},
  {{"job_id": "456", "rating": "Low", "bullets": ["Python and SQL skills match", "No healthcare domain experience required by role", "Missing 5+ years of people management required"]}}
]

Return ONLY the JSON array, no other text."""


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
        search_url = get_search_url()
        print("Navigating to LinkedIn job search...")
        page.goto(search_url, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)  # Wait for JS rendering

        # Step 1: Collect all job IDs first (lightweight, no DOM mutation)
        job_ids = page.eval_on_selector_all(
            "[data-job-id]",
            "els => els.map(el => el.getAttribute('data-job-id')).filter(Boolean)"
        )

        if not job_ids:
            print("WARNING: No job cards found. Cookie may be expired or page layout changed.")
            page.screenshot(path="debug_screenshot.png")
            browser.close()
            return []

        print(f"Found {len(job_ids)} job cards")

        # Step 2: For each job ID, re-locate the card in the DOM, extract info, click, get description
        for i, job_id in enumerate(job_ids):
            job_url = f"https://www.linkedin.com/jobs/view/{job_id}/"

            try:
                # Re-query the card fresh each time (DOM may have re-rendered)
                card = page.query_selector(f'[data-job-id="{job_id}"]')
                if not card:
                    print(f"  [{i+1}/{len(job_ids)}] Card not found for job {job_id}, skipping")
                    continue

                # Extract title from <strong> inside link
                title_el = card.query_selector('a[href*="/jobs/view/"] strong')
                title = title_el.inner_text().strip() if title_el else "Unknown"

                # Extract company
                company_el = card.query_selector(".artdeco-entity-lockup__subtitle")
                company = company_el.inner_text().strip() if company_el else "Unknown"

                # Extract card metadata (location, salary, etc.)
                card_meta = card.eval_on_selector_all("li", "els => els.map(el => el.innerText.trim())")

                location = card_meta[0] if card_meta else "Unknown"

                print(f"  [{i+1}/{len(job_ids)}] Loading: {title} @ {company}")

                # Click the card to load job description in the detail panel
                card.click()
                page.wait_for_timeout(random.randint(2000, 4000))

                # Wait for the detail panel to load
                page.wait_for_selector("#job-details", timeout=10000)

                # Extract full description
                description = page.eval_on_selector(
                    "#job-details", "el => el.innerText.trim()"
                ) or ""

                # Extract salary from detail panel badges
                salary_from_detail = page.evaluate("""() => {
                    const spans = document.querySelectorAll(
                        '.job-details-jobs-unified-top-card__job-insight-view-model-secondary span'
                    );
                    for (const s of spans) {
                        const t = s.innerText.trim();
                        if (t.includes('$') && t.toLowerCase().includes('/yr')) return t;
                    }
                    return '';
                }""")

            except Exception as e:
                print(f"    Error processing job {job_id}: {e}")
                title = title if 'title' in dir() else "Unknown"
                company = company if 'company' in dir() else "Unknown"
                location = location if 'location' in dir() else "Unknown"
                card_meta = card_meta if 'card_meta' in dir() else []
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

    # Filter out jobs with no title and no description (completely failed scrapes)
    valid_jobs = [j for j in jobs if j["job_title"] != "Unknown" or j["description"]]
    if len(valid_jobs) < len(jobs):
        print(f"  Filtered out {len(jobs) - len(valid_jobs)} jobs with no data")
    return valid_jobs


def generate_comments(jobs: list[dict], api_key: str) -> list[dict]:
    """Use Gemini 2.5 Flash to evaluate all jobs in a single batched request."""
    client = genai.Client(api_key=api_key)

    # Build the jobs block for the prompt
    jobs_block_parts = []
    for job in jobs:
        jobs_block_parts.append(
            f"[Job ID: {job['job_id']}]\n"
            f"Title: {job['job_title']}\n"
            f"Company: {job['company']}\n"
            f"Description:\n{job['description'][:2000]}\n"
        )
    jobs_block = "\n---\n".join(jobs_block_parts)

    prompt = GEMINI_BATCH_PROMPT_TEMPLATE.format(
        resume=CANDIDATE_RESUME,
        jobs_block=jobs_block,
    )

    print(f"  Sending 1 batched request for {len(jobs)} jobs...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw_text = response.text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        results = json.loads(raw_text)
        results_map = {r["job_id"]: r for r in results}

        for job in jobs:
            result = results_map.get(job["job_id"])
            if result:
                rating = result.get("rating", "Unknown")
                bullets = result.get("bullets", [])
                # Format as: "Rating: High\n- bullet1\n- bullet2\n- bullet3"
                bullet_text = "\n".join(f"- {b}" for b in bullets)
                job["comment"] = f"Rating: {rating}\n{bullet_text}"
            else:
                job["comment"] = "Rating: Unknown\n- No analysis returned"
                print(f"  WARNING: No result for job {job['job_id']} ({job['job_title']})")

    except json.JSONDecodeError as e:
        print(f"  Gemini JSON parse error: {e}")
        print(f"  Raw response: {raw_text[:500]}")
        for job in jobs:
            job["comment"] = "Error: failed to parse Gemini response"
    except Exception as e:
        print(f"  Gemini API error: {e}")
        for job in jobs:
            job["comment"] = "Error generating comment"

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
        # Extract bullet lines from comment (lines starting with "- ")
        bullet_lines = [line.strip() for line in job["comment"].split("\n") if line.strip().startswith("- ")]
        bullets_html = "".join(f"<li>{b[2:]}</li>" for b in bullet_lines)

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
                <strong>[{rating}]</strong>
                <ul style="margin: 4px 0; padding-left: 18px; font-size: 13px;">{bullets_html}</ul>
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
            <th style="padding: 12px; border: 1px solid #ddd; text-align: left; width: 45%;">Resume Match Analysis</th>
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
