class StakeholderAnalyzer:

    @staticmethod
    def extract(data: dict):
        report = data.get("stakeholder_report", {})

        return {
            "beneficiaries": report.get("beneficiaries", []),
            "constraints": report.get("constraints", []),
            "opportunities": report.get("opportunities", [])
        }