"""Audited access decisions, independent of route planning and booking weeks."""
import re

# Explicit review hold from Conor's audit, not a generic restriction on large sites.
INTERNAL_REVIEW_REFERENCES = {"BRAU0000"}


def classify_access(reason, customer_failures, metro_failures=0, reference=""):
    """Return a clear rule decision, or None for genuinely unrecognised prose.

    Failure counts come from Salesforce, never from phrases such as 'rang twice'.
    A failed key with a usable resident-entry alternative remains a routine retry.
    """
    text = str(reason or "").strip().lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text)

    def answer(decision, category, why, action="", **extra):
        return dict(Decision=decision, **{"Reason Category": category,
            "Decision Reason": why, "Recommended Client Action": action,
            "Decision Source": "Audited access rules", **extra})

    def client(category, why, action):
        return answer("CLIENT_ACCESS_REQUIRED", category, why, action)

    if customer_failures >= 2:
        return client("REPEATED_CUSTOMER_FAILURE", "Two or more customer failures are recorded.",
                      "Arrange an access appointment with the occupier or site manager.")
    if reference in INTERNAL_REVIEW_REFERENCES:
        return answer("IGNORE", "INTERNAL_REVIEW", "Conor requested internal investigation before replanning.")
    if re.search(r"planner|planstudio|plan studio|blueprint|drawing|same for all|same for all for", text):
        return answer("IGNORE", "INTERNAL_TECHNICAL", "Resolve the drawing, system or instruction issue before replanning.")
    if re.search(r"no power|not clear on what building|unclear which building|isn't clear on what", text):
        return answer("IGNORE", "INTERNAL_INSTRUCTION", "Confirm the survey instruction or operational issue before replanning.")
    if re.search(r"roadworks|road works", text):
        return answer("RETRY", "TEMPORARY_ROADWORKS", "Temporary roadworks: check route access and retry.")
    if re.search(r"boarded|construction|renovation|possession", text):
        return client("SITE_CONDITION", "Building works or boarding prevent normal access.",
                      "Confirm that the survey can proceed and arrange safe access.")
    if re.search(r"appointment|need.*booking|escort|staff.*present|notice is required|lack of contact|contact number", text):
        return client("APPOINTMENT_REQUIRED", "Access requires an appointment, notice, staff or an escort.",
                      "Provide a site contact and agree an appointment and any escort requirements.")
    if re.search(r"refus|denied|den(y|i)ed|turned away|wouldn'?t|won'?t allow|wont allow|won't.*let|would not.*let|said no|said 'no'|go away|police|wasn'?t notified|not been notified|no prior notice|lack of information|uncooperative|not.*cooperative|not allowing|wont come|won't come", text):
        return client("ACCESS_REFUSED", "A resident declined access or requires confirmation/notice.",
                      "Explain the survey, provide notice and confirm consent and an access appointment.")
    if re.search(r"cannot find|can't find|not clear on where.*entrance", text):
        return client("ADDRESS_OR_ENTRANCE", "The building or entrance could not be located or confirmed.",
                      "Confirm the building and entrance with access instructions and a marked plan or photo.")
    if re.search(r"office.*monday|office.*friday", text):
        return answer("RETRY_WITH_CONSTRAINT", "NO_ANSWER", "Retry when the office is staffed.",
                      **{"Preferred Weekdays": ["Monday", "Friday"], "Required Weekdays": ["Monday", "Friday"]})
    # Explicit lack of an alternative takes precedence over a no-answer phrase.
    blocked = re.search(r"no (?:working )?intercom|no (?:individual )?(?:buttons|doorbells)|no where.*call|nowhere.*call|no way.*(?:get|enter)|no usable|out of service|manually locked|main gate locked|gate.*key pad|yale key|required.*key|don't have.*keys.*car|no.*key.*car|inside and out|first hallway", text)
    broken = re.search(r"(?:bell|bells|intercom).*(?:don't work|doesn't work|do not work|not work|broken)", text)
    knocked = bool(re.search(r"knock", text))
    resident_attempt = bool(re.search(r"no answer|no response|no one home|no one in|none in|not home|at work|no residents|did not answer|did not get an answer|no one else is in|unable to get responses|gotten no|ad no answer|not in all day|won't be in", text))
    key_issue = bool(re.search(r"key|fob|locked|keypad", text))
    if blocked or (broken and not knocked) or (key_issue and not resident_attempt):
        return client("ENTRY_SYSTEM", "A usable entry route, key or access system is needed.",
                      "Provide a working key/fob/code or arrange someone to meet the surveyor.")
    if metro_failures and not customer_failures:
        return answer("RETRY", "METRO_OPERATIONAL", "Operational failure may be retried on any available weekday.")
    if resident_attempt or text in {"", "other", "unable to get access", "unable to gain access", "no access", "no answer at door"}:
        return answer("RETRY", "NO_ANSWER", "No answer or generic access failure below the customer limit; retry on a different weekday.")
    return None
