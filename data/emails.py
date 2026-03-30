"""
data/emails.py — Synthetic labeled email dataset for all 3 tasks.
Each email has ground truth labels for category, priority, and routing.
"""

EMAILS = [
    # ── SPAM ──────────────────────────────────────────────────────────────
    {
        "email_id": "e001",
        "subject": "🎉 You've won $1,000,000! Claim now!",
        "body": "Congratulations! You have been selected as the lucky winner of our international lottery. Click the link below to claim your prize immediately. Act fast, offer expires soon! www.total-scam-link.xyz",
        "sender": "winner@lottery-scam.xyz",
        "sender_name": "Lottery International",
        "timestamp": "2024-03-15T08:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "spam",
            "priority": "low",
            "route_to": "trash",
            "reply_needed": False
        }
    },
    {
        "email_id": "e002",
        "subject": "LIMITED OFFER: Buy Viagra cheap!!!",
        "body": "Best prices on all medications. No prescription needed. Order now and get 70% off. Free shipping worldwide. Click here to order: www.cheap-meds-scam.com",
        "sender": "deals@pharma-scam.net",
        "sender_name": "Pharma Deals",
        "timestamp": "2024-03-15T08:15:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "spam",
            "priority": "low",
            "route_to": "trash",
            "reply_needed": False
        }
    },
    # ── URGENT ────────────────────────────────────────────────────────────
    {
        "email_id": "e003",
        "subject": "CRITICAL: Production database is DOWN",
        "body": "Hi team,\n\nOur production database went offline at 14:32 UTC. All services are affected. Users cannot log in. Revenue impact is approximately $10,000 per minute.\n\nI need all hands on deck immediately. Please join the war room: meet.google.com/xyz\n\nThis is a P0 incident.\n\n— Ravi, SRE Lead",
        "sender": "ravi.sre@company.com",
        "sender_name": "Ravi Kumar",
        "timestamp": "2024-03-15T14:35:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "urgent",
            "priority": "critical",
            "route_to": "escalate",
            "reply_needed": True,
            "ideal_reply_keywords": ["joining", "on it", "immediately", "war room", "help"]
        }
    },
    {
        "email_id": "e004",
        "subject": "Server security breach detected - immediate action required",
        "body": "Dear Admin,\n\nOur monitoring system has detected unauthorized access attempts on server prod-us-east-1. Multiple failed login attempts from IP 192.168.1.45 followed by a successful login at 03:12 AM. Unusual data transfer of 2.3GB initiated.\n\nPlease investigate immediately and consider taking the server offline.\n\n— Security Bot",
        "sender": "alerts@security-monitor.company.com",
        "sender_name": "Security Monitor",
        "timestamp": "2024-03-15T03:20:00Z",
        "thread_length": 1,
        "has_attachment": True,
        "ground_truth": {
            "category": "urgent",
            "priority": "critical",
            "route_to": "escalate",
            "reply_needed": True,
            "ideal_reply_keywords": ["investigating", "security team", "taking action", "offline", "breach"]
        }
    },
    # ── SUPPORT ───────────────────────────────────────────────────────────
    {
        "email_id": "e005",
        "subject": "Cannot login to my account - please help",
        "body": "Hello Support,\n\nI have been trying to login to my account for the past 2 hours but keep getting 'Invalid password' error. I've tried resetting my password 3 times but the reset email never arrives.\n\nMy username is john.doe@gmail.com. Please help urgently as I need to access my files for a presentation tomorrow morning.\n\nThanks,\nJohn Doe",
        "sender": "john.doe@gmail.com",
        "sender_name": "John Doe",
        "timestamp": "2024-03-15T18:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "support",
            "priority": "high",
            "route_to": "support",
            "reply_needed": True,
            "ideal_reply_keywords": ["password reset", "check spam", "help", "account", "team"]
        }
    },
    {
        "email_id": "e006",
        "subject": "App crashes when I upload files larger than 10MB",
        "body": "Hi,\n\nI'm experiencing a bug in your app. Whenever I try to upload a file larger than 10MB, the app crashes completely and I lose all my unsaved work. This has happened 5 times today.\n\nI'm using version 3.2.1 on macOS Ventura 13.4. Please fix this ASAP.\n\nBest,\nSarah",
        "sender": "sarah.m@outlook.com",
        "sender_name": "Sarah Mitchell",
        "timestamp": "2024-03-15T11:30:00Z",
        "thread_length": 2,
        "has_attachment": False,
        "ground_truth": {
            "category": "support",
            "priority": "high",
            "route_to": "support",
            "reply_needed": True,
            "ideal_reply_keywords": ["bug", "investigating", "version", "workaround", "fix"]
        }
    },
    # ── BILLING ───────────────────────────────────────────────────────────
    {
        "email_id": "e007",
        "subject": "Incorrect charge on my invoice #INV-2024-0312",
        "body": "To Whom It May Concern,\n\nI received invoice #INV-2024-0312 for $2,450 but I should have been charged $1,200 based on our contract for the Basic Plan. It appears I have been billed for the Enterprise Plan.\n\nPlease correct this immediately and issue a revised invoice. I will not pay until this is resolved.\n\n— Michael Torres",
        "sender": "m.torres@acmecorp.com",
        "sender_name": "Michael Torres",
        "timestamp": "2024-03-15T09:45:00Z",
        "thread_length": 1,
        "has_attachment": True,
        "ground_truth": {
            "category": "billing",
            "priority": "high",
            "route_to": "billing",
            "reply_needed": True,
            "ideal_reply_keywords": ["invoice", "billing team", "review", "correct", "apologies"]
        }
    },
    {
        "email_id": "e008",
        "subject": "Question about annual plan pricing",
        "body": "Hi,\n\nI'm currently on the monthly plan at $49/month. I'm considering switching to the annual plan. Could you confirm the annual pricing and whether I get any discount for upgrading mid-cycle?\n\nAlso, does the annual plan include all the same features as the monthly plan?\n\nThank you!",
        "sender": "priya.s@startup.io",
        "sender_name": "Priya Sharma",
        "timestamp": "2024-03-15T10:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "billing",
            "priority": "medium",
            "route_to": "billing",
            "reply_needed": True,
            "ideal_reply_keywords": ["annual", "discount", "pricing", "features", "plan"]
        }
    },
    # ── NEWSLETTER ────────────────────────────────────────────────────────
    {
        "email_id": "e009",
        "subject": "Your Weekly Product Digest — March 15, 2024",
        "body": "Hello!\n\nHere's what's new this week:\n• New dashboard feature launched\n• Performance improvements for API users\n• Upcoming webinar: AI in productivity — March 22nd\n• Blog post: How our users save 3 hours/week\n\nVisit our blog for the full stories. Have a great weekend!\n\n— The Product Team\n\nUnsubscribe | Manage Preferences",
        "sender": "newsletter@product.company.com",
        "sender_name": "Product Team",
        "timestamp": "2024-03-15T07:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "newsletter",
            "priority": "low",
            "route_to": "inbox",
            "reply_needed": False
        }
    },
    {
        "email_id": "e010",
        "subject": "TechCrunch Daily: Top stories in AI this week",
        "body": "Good morning,\n\nToday's top stories:\n1. OpenAI announces GPT-5 development\n2. Google DeepMind achieves new robotics milestone\n3. EU AI Act implementation timeline revealed\n4. Startup funding in AI sector hits record high\n\nRead the full stories at techcrunch.com.\n\nUnsubscribe from this list",
        "sender": "daily@techcrunch.com",
        "sender_name": "TechCrunch",
        "timestamp": "2024-03-15T06:30:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "newsletter",
            "priority": "low",
            "route_to": "inbox",
            "reply_needed": False
        }
    },
    # ── NORMAL ────────────────────────────────────────────────────────────
    {
        "email_id": "e011",
        "subject": "Meeting notes from yesterday's standup",
        "body": "Hi all,\n\nHere are the notes from yesterday's standup:\n\nCompleted:\n- API rate limiting implementation (Dev team)\n- Q1 report draft (Finance)\n\nIn Progress:\n- UI redesign (Design team - ETA: Friday)\n- Customer onboarding flow update\n\nBlockers:\n- Need legal review on new ToS before launch\n\nNext standup: Tomorrow 10 AM.\n\n— Aisha",
        "sender": "aisha.pm@company.com",
        "sender_name": "Aisha Johnson",
        "timestamp": "2024-03-15T09:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "normal",
            "priority": "low",
            "route_to": "inbox",
            "reply_needed": False
        }
    },
    {
        "email_id": "e012",
        "subject": "HR: Reminder — Performance review submissions due Friday",
        "body": "Dear Team,\n\nThis is a reminder that self-assessment forms for the Q1 performance review are due this Friday, March 17th by 5 PM.\n\nPlease complete your self-assessment at: hr-portal.company.com/review\n\nIf you have any questions, contact the HR team.\n\nBest,\nHR Department",
        "sender": "hr@company.com",
        "sender_name": "HR Department",
        "timestamp": "2024-03-15T09:30:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "ground_truth": {
            "category": "hr",
            "priority": "medium",
            "route_to": "hr",
            "reply_needed": False
        }
    },
]

# Task-specific email subsets
TASK_EASY_EMAILS = EMAILS[:6]    # Mix of spam, urgent, normal — binary classification
TASK_MEDIUM_EMAILS = EMAILS[4:]  # Support, billing, hr, normal — routing challenge
TASK_HARD_EMAILS = [e for e in EMAILS if e["ground_truth"]["reply_needed"]]  # Must draft reply