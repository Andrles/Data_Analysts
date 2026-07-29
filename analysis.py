from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
DATA_PATH = Path("data/online_shoppers_intention.csv")


def load_data() -> pd.DataFrame:
    """Загружает набор данных UCI при первом запуске."""
    if not DATA_PATH.exists():
        print("Скачиваю данные из UCI Machine Learning Repository...")
        DATA_PATH.parent.mkdir(exist_ok=True)
        with urlopen(DATA_URL) as response:
            archive = ZipFile(BytesIO(response.read()))
            with archive.open("online_shoppers_intention.csv") as source:
                DATA_PATH.write_bytes(source.read())
    return pd.read_csv(DATA_PATH)


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = load_data()
    conversion = df["Revenue"].mean()

    print(f"Количество сессий: {len(df):,}")
    print(f"Общая конверсия: {conversion:.1%}")
    print("\nКонверсия по типу посетителя:")
    print(df.groupby("VisitorType")["Revenue"].agg(["size", "mean"]).round(3))

    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = df.groupby("Month", as_index=False)["Revenue"].mean()
    monthly["Month"] = pd.Categorical(monthly["Month"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("Month")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=monthly, x="Month", y="Revenue", color="#4C78A8")
    plt.title("Конверсия в покупку по месяцам")
    plt.xlabel("Месяц")
    plt.ylabel("Конверсия")
    plt.ylim(0, 0.3)
    plt.tight_layout()
    plt.savefig("conversion_by_month.png", dpi=150)
    print("\nГрафик сохранён в conversion_by_month.png")


if __name__ == "__main__":
    main()
