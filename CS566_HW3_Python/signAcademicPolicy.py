def sign_academic_honesty_policy(name: str, uni: str):
    """
    Prints an academic honesty statement or prints an error if not properly signed.
    """

    if name == "full_name" or uni == "stu_id":
        statement_str = "ERROR: Academic Honesty Policy agreement was not signed."
    else:
        statement_str = (
            f"I, {name} ({uni}), \n"
            "certify that I have read and agree to the Code of Academic Integrity."
        )

    header = "\n\n***********************\n"
    footer = "\n***********************\n\n"

    print(f"{header}{statement_str}{footer}")
