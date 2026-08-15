"""Static source contracts for the Home Credit Default Risk files."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TableContract:
    file_name: str
    table_name: str
    grain: str
    primary_key: tuple[str, ...]
    required_columns: tuple[str, ...]
    min_columns: int
    required_file: bool = True
    foreign_keys: tuple[tuple[str, str], ...] = ()
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


DATASET_CONTRACTS: tuple[TableContract, ...] = (
    TableContract(
        file_name="application_train.csv",
        table_name="application_train",
        grain="one current loan application per labelled applicant",
        primary_key=("SK_ID_CURR",),
        required_columns=(
            "SK_ID_CURR",
            "TARGET",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
        ),
        min_columns=100,
        notes="Controlling labelled modeling population.",
    ),
    TableContract(
        file_name="application_test.csv",
        table_name="application_test",
        grain="one current loan application per unlabelled applicant",
        primary_key=("SK_ID_CURR",),
        required_columns=("SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"),
        min_columns=100,
    ),
    TableContract(
        file_name="bureau.csv",
        table_name="bureau",
        grain="one previous external credit account",
        primary_key=("SK_ID_BUREAU",),
        required_columns=("SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT"),
        min_columns=15,
        foreign_keys=(("SK_ID_CURR", "application.SK_ID_CURR"),),
    ),
    TableContract(
        file_name="bureau_balance.csv",
        table_name="bureau_balance",
        grain="one relative month per previous external credit account",
        primary_key=("SK_ID_BUREAU", "MONTHS_BALANCE"),
        required_columns=("SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"),
        min_columns=3,
        foreign_keys=(("SK_ID_BUREAU", "bureau.SK_ID_BUREAU"),),
    ),
    TableContract(
        file_name="previous_application.csv",
        table_name="previous_application",
        grain="one historical Home Credit application",
        primary_key=("SK_ID_PREV",),
        required_columns=("SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS"),
        min_columns=30,
        foreign_keys=(("SK_ID_CURR", "application.SK_ID_CURR"),),
    ),
    TableContract(
        file_name="POS_CASH_balance.csv",
        table_name="pos_cash_balance",
        grain="one relative month per historical POS or cash loan",
        primary_key=("SK_ID_PREV", "MONTHS_BALANCE"),
        required_columns=("SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD"),
        min_columns=8,
        foreign_keys=(
            ("SK_ID_CURR", "application.SK_ID_CURR"),
            ("SK_ID_PREV", "previous_application.SK_ID_PREV"),
        ),
    ),
    TableContract(
        file_name="credit_card_balance.csv",
        table_name="credit_card_balance",
        grain="one relative month per historical credit-card account",
        primary_key=("SK_ID_PREV", "MONTHS_BALANCE"),
        required_columns=("SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE"),
        min_columns=20,
        foreign_keys=(
            ("SK_ID_CURR", "application.SK_ID_CURR"),
            ("SK_ID_PREV", "previous_application.SK_ID_PREV"),
        ),
    ),
    TableContract(
        file_name="installments_payments.csv",
        table_name="installments_payments",
        grain="one observed payment against a historical scheduled installment",
        primary_key=(),
        required_columns=(
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ),
        min_columns=8,
        foreign_keys=(
            ("SK_ID_CURR", "application.SK_ID_CURR"),
            ("SK_ID_PREV", "previous_application.SK_ID_PREV"),
        ),
        notes="Partial payments can create multiple rows for an installment; do not assert uniqueness.",
    ),
    TableContract(
        file_name="HomeCredit_columns_description.csv",
        table_name="column_descriptions",
        grain="one documented source-column description",
        primary_key=(),
        required_columns=("Table", "Row", "Description"),
        min_columns=4,
        required_file=False,
        notes="May require a legacy single-byte encoding.",
    ),
    TableContract(
        file_name="sample_submission.csv",
        table_name="sample_submission",
        grain="one unlabelled application prediction placeholder",
        primary_key=("SK_ID_CURR",),
        required_columns=("SK_ID_CURR", "TARGET"),
        min_columns=2,
        required_file=False,
    ),
)


def contract_by_file_name(file_name: str) -> TableContract:
    for contract in DATASET_CONTRACTS:
        if contract.file_name.casefold() == file_name.casefold():
            return contract
    raise KeyError(f"No dataset contract for {file_name}")
