from abc import ABC, abstractmethod


class BaseDomainClassifier(ABC):
    @abstractmethod
    def classify(self, text: str) -> tuple[str, float]:
        pass


class KeywordDomainClassifier(BaseDomainClassifier):
    def __init__(self, domain_keywords: dict[str, list[str]]) -> None:
        self.domain_keywords = domain_keywords

    def classify(self, text: str) -> tuple[str, float]:
        text_lower = (text or "").lower()
        if not self.domain_keywords:
            return "general", 0.0

        scores = {domain: 0 for domain in self.domain_keywords}
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                scores[domain] += text_lower.count(str(keyword).lower())

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]
        total_score = sum(scores.values())

        if best_score <= 0 or total_score <= 0:
            return "general", 0.0

        confidence = round(best_score / total_score, 4)
        return best_domain, confidence
