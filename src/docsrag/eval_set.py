"""A small labelled set: question -> the section that should be retrieved."""

EVAL_SET: list[tuple[str, str]] = [
    (
        "My laptop says the office printer is offline. What do I do?",
        "Printer shows offline",
    ),
    ("Paper is stuck inside the printer", "Clearing a paper jam"),
    (
        "How do I add the office printer to a new laptop?",
        "Adding a printer to a new laptop",
    ),
    ("When should I order a new toner cartridge?", "Toner and ink levels"),
    (
        "Can I put my personal phone on the staff Wi-Fi?",
        "Connecting a device to staff Wi-Fi",
    ),
    (
        "Wi-Fi is connected but websites will not load",
        "Connected to Wi-Fi but no internet",
    ),
    (
        "Why is the internet so slow when the whole class is online?",
        "Slow internet in the computer lab",
    ),
    (
        "In what order should I restart the router and access points?",
        "Restarting the router and access points",
    ),
    ("A teacher forgot her email password", "Resetting a staff password"),
    ("How do I turn on two-step verification?", "Turning on two-step verification"),
    ("How do I create an account for a new teacher?", "Setting up a new staff member"),
    (
        "What happens to the account when a staff member leaves?",
        "When a staff member leaves",
    ),
    ("How do I publish a news post on the school website?", "Publishing a news post"),
    ("Can we post photos of learners with their names?", "Photos and learner privacy"),
    ("Who approves website content before it goes live?", "Who approves content"),
    ("How often is the office computer backed up?", "What is backed up and when"),
    ("I deleted a file from Google Drive by mistake", "Restoring a deleted file"),
    (
        "Can I send learner results to a WhatsApp group?",
        "Keeping learner records private",
    ),
    ("Why should marks be entered as numbers only?", "Results spreadsheets"),
    ("What should I check before a lesson in the lab?", "Checklist before a lesson"),
    ("A lab computer keeps beeping and restarting", "A computer will not start"),
    ("Can learners use USB drives on lab computers?", "Viruses and USB drives"),
]
