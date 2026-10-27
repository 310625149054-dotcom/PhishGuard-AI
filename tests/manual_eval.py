import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import predict_email

phishing_emails = [
    # Phishing 1: Classic financial urgency with IP URL and credential trigger
    """Subject: URGENT: Your PayPal Account Has Been Suspended!
Dear User, We detected unauthorized logins from an unknown device. Verify your password immediately at http://192.168.1.1/paypal/verify.php to avoid account termination.""",

    # Phishing 2: Microsoft 365 credential harvesting with shortener link
    """Subject: Action Required: Your Office 365 Password Expires Today
All employees must update their corporate credentials immediately to retain mailbox access. Click the link below to confirm your password:
http://bit.ly/m365-password-update
IT Helpdesk Support Team""",

    # Phishing 3: Delivery invoice lure with suspicious TLD
    """Subject: Overdue Invoice #94821 - Immediate Payment Required
Please find your pending billing statement attached. Click here to confirm payment details: http://billing-update.xyz/invoice?id=94821
If not paid within 24 hours, legal collection action will proceed."""
]

safe_emails = [
    # Safe 1: Workplace project status report
    """Subject: Project Falcon: Sprint 14 Release Notes
Hi team,
The latest staging build is ready for review. Please check the test coverage report on the company intranet: https://internal.company.com/builds/sprint14
Great job delivering the authentication module on schedule.
Best regards, Alex""",

    # Safe 2: Campus academic scheduling announcement
    """Subject: Computer Science Faculty Seminar - Thursday 3:00 PM
Dear Students,
You are invited to this week's seminar on Natural Language Processing and LLMs. The lecture will be held in Auditorium 2. Refreshments will be provided after the Q&A session.
Regards, Department of Computer Science""",

    # Safe 3: Casual personal lunch coordination
    """Subject: Lunch tomorrow?
Hey Sarah, are we still meeting at the campus cafe around 12:30 tomorrow? Let me know if that time still works for you!"""
]

print("=== TESTING PHISHING EXAMPLES ===")
for idx, text in enumerate(phishing_emails, 1):
    res = predict_email(text)
    print(f"Phishing #{idx}: Label={res['label']}, Confidence={res['confidence']}%, Risk={res['risk_level']}, Red Flags={len(res['red_flags'])}")
    for f in res['red_flags']:
        print(f"   - {f}")

print("\n=== TESTING SAFE EXAMPLES ===")
for idx, text in enumerate(safe_emails, 1):
    res = predict_email(text)
    print(f"Safe #{idx}: Label={res['label']}, Confidence={res['confidence']}%, Risk={res['risk_level']}, Red Flags={len(res['red_flags'])}")
    for f in res['red_flags']:
        print(f"   - {f}")
