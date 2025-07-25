import re


def apply_guardrails(answer: str, lang: str = 'english') -> str:
    """
    Apply guardrails to ensure the response is safe, appropriate, and suitable for a multilingual educational chatbot.
    Args:
        answer (str): The generated response to check.
        lang (str): The language of the response (e.g., 'english', 'bengali', 'spanish', 'hindi').
    Returns:
        str: The safe response or a warning if inappropriate content is detected.
    """
    # Define inappropriate content for multiple languages
    inappropriate_content = {
        'english': [
            r'\b(badword|offensive|hate|insult|curse|swear|profanity|inappropriate|explicit|vulgar|violence|misinformation)\b',
            r'\b(sex|drugs|alcohol|weapon|suicide|self-harm)\b'
        ],
        'bengali': [
            r'\b(খারাপ শব্দ|অপমান|ঘৃণা|অভিশাপ|অশ্লীল|হিংসা|ভুল তথ্য)\b',
            # bad word, insult, hate, curse, vulgar, violence, misinformation
            r'\b(যৌন|মাদক|মদ্যপান|অস্ত্র|আত্মহত্যা|আত্ম-ক্ষতি)\b'  # sex, drugs, alcohol, weapon, suicide, self-harm
        ],
    }

    # Define education-specific inappropriate content
    education_sensitive = {
        'english': [
            r'\b(cheating|plagiarism|fake answers|off-topic|non-educational)\b',
            r'\b(discrimination|bullying|harassment)\b'
        ],
        'bengali': [
            r'\b(প্রতারণা|চুরি করা|ভুয়া উত্তর|বিষয়ের বাইরে|অশিক্ষামূলক)\b',
            # cheating, plagiarism, fake answers, off-topic, non-educational
            r'\b(বৈষম্য|হয়রানি|ধমকানো)\b'  # discrimination, harassment, bullying
        ],
    }

    # Normalize language input
    lang_key = lang.lower()
    if lang_key not in inappropriate_content:
        lang_key = 'english'  # Default to English if language is unsupported

    # Check for inappropriate content
    patterns = inappropriate_content.get(lang_key, inappropriate_content['english'])
    for pattern in patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return f"⚠️ Inappropriate content detected. Please rephrase your query to align with educational guidelines."

    # Check for education-specific sensitive content
    sensitive_patterns = education_sensitive.get(lang_key, education_sensitive['english'])
    for pattern in sensitive_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return f"⚠️ The response contains content unsuitable for an educational context. Please try a different question or contact a teacher for assistance."

    # If no issues are found, return the original answer
    return answer
