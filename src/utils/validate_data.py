from typing import Tuple, List
import pandas as pd

# Try Great Expectations; fall back to lightweight pandas checks if unavailable.
try:
    from great_expectations.dataset import PandasDataset  # type: ignore
    GE_AVAILABLE = True
except Exception:
    GE_AVAILABLE = False


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    if GE_AVAILABLE:
        print("🔍 Starting data validation with Great Expectations...")
        ge_df = PandasDataset(df)
    else:
        print("⚠️ Great Expectations not available; running simplified pandas checks.")
    
    required_columns = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    failed_expectations: List[str] = []

    if GE_AVAILABLE:
        # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
        print("   📋 Validating schema and required columns...")
        ge_df.expect_column_to_exist("customerID")
        ge_df.expect_column_values_to_not_be_null("customerID")
        ge_df.expect_column_to_exist("gender")
        ge_df.expect_column_to_exist("Partner")
        ge_df.expect_column_to_exist("Dependents")
        ge_df.expect_column_to_exist("PhoneService")
        ge_df.expect_column_to_exist("InternetService")
        ge_df.expect_column_to_exist("Contract")
        ge_df.expect_column_to_exist("tenure")
        ge_df.expect_column_to_exist("MonthlyCharges")
        ge_df.expect_column_to_exist("TotalCharges")

        # === BUSINESS LOGIC VALIDATION ===
        print("   💼 Validating business logic constraints...")
        ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
        ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
        ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
        ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
        ge_df.expect_column_values_to_be_in_set(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        ge_df.expect_column_values_to_be_in_set(
            "InternetService", ["DSL", "Fiber optic", "No"]
        )

        # === NUMERIC RANGE VALIDATION ===
        print("   📊 Validating numeric ranges and business constraints...")
        ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
        ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
        ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
        ge_df.expect_column_values_to_not_be_null("tenure")
        ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

        # === DATA CONSISTENCY CHECKS ===
        print("   🔗 Validating data consistency...")
        ge_df.expect_column_pair_values_A_to_be_greater_than_B(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95
        )

        print("   ⚙️  Running complete validation suite...")
        results = ge_df.validate()

        failed_expectations = [
            r["expectation_config"]["expectation_type"]
            for r in results["results"] if not r["success"]
        ]
        total_checks = len(results["results"])
        passed_checks = total_checks - len(failed_expectations)

        if results["success"]:
            print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
            return True, []
        else:
            print(f"❌ Data validation FAILED: {len(failed_expectations)}/{total_checks} checks failed")
            print(f"   Failed expectations: {failed_expectations}")
            return False, failed_expectations

    # === Lightweight fallback validation (no GE) ===
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        failed_expectations.append("missing_required_columns")
        return False, failed_expectations

    # Basic value checks
    if not df["gender"].isin(["Male", "Female"]).all():
        failed_expectations.append("gender_invalid")
    for col in ["Partner", "Dependents", "PhoneService"]:
        if not df[col].isin(["Yes", "No"]).all():
            failed_expectations.append(f"{col}_invalid")
    if not df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all():
        failed_expectations.append("contract_invalid")
    if not df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all():
        failed_expectations.append("internet_invalid")

    # Coerce numeric columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Numeric ranges
    if (df["tenure"] < 0).any() or (df["tenure"] > 120).any():
        failed_expectations.append("tenure_range")
    if (df["MonthlyCharges"] < 0).any() or (df["MonthlyCharges"] > 200).any():
        failed_expectations.append("monthly_charges_range")
    if (df["TotalCharges"] < 0).any():
        failed_expectations.append("total_charges_negative")

    if failed_expectations:
        print(f"❌ Data validation FAILED (fallback): {failed_expectations}")
        return False, failed_expectations

    print("✅ Data validation PASSED (fallback checks)")
    return True, []
