"""Build olist_simple_orders.csv: 1,000 real Olist orders in ONE table, for the guided prompt lab.
Source: Olist Brazilian E-Commerce Public Dataset (Kaggle, CC BY-NC-SA 4.0). Run: python make_simple_orders.py OLIST_DIR"""
import sys
from pathlib import Path
import pandas as pd

F = Path(sys.argv[1])
o = pd.read_csv(F / "olist_orders_dataset.csv")
for c in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
    o[c] = pd.to_datetime(o[c])
r = pd.read_csv(F / "olist_order_reviews_dataset.csv", parse_dates=["review_answer_timestamp"])
r = r.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
i = pd.read_csv(F / "olist_order_items_dataset.csv")
p = pd.read_csv(F / "olist_products_dataset.csv")
t = pd.read_csv(F / "product_category_name_translation.csv")
c = pd.read_csv(F / "olist_customers_dataset.csv")

d = o[(o["order_status"] == "delivered") & o["order_delivered_customer_date"].notna()]
d = d.merge(r[["order_id", "review_score"]], on="order_id").merge(c[["customer_id", "customer_state"]], on="customer_id")
items = i.merge(p[["product_id", "product_category_name"]], on="product_id", how="left").merge(t, on="product_category_name", how="left")
per = items.groupby("order_id").agg(n_items=("order_item_id", "size"), price=("price", "sum"), freight=("freight_value", "sum"),
                                     category=("product_category_name_english", "first")).reset_index()
d = d.merge(per, on="order_id")
d["delivery_days"] = (d["order_delivered_customer_date"] - d["order_purchase_timestamp"]).dt.days
d["days_late"] = (d["order_delivered_customer_date"].dt.normalize() - d["order_estimated_delivery_date"]).dt.days
out = d.sample(1000, random_state=2026).sort_values("order_purchase_timestamp")
out = pd.DataFrame({
    "order_id": out["order_id"].str[:8],
    "purchase_date": out["order_purchase_timestamp"].dt.date,
    "customer_state": out["customer_state"],
    "category": out["category"],
    "n_items": out["n_items"],
    "price": out["price"].round(2),
    "freight": out["freight"].round(2),
    "delivery_days": out["delivery_days"],
    "days_late": out["days_late"],
    "review_score": out["review_score"],
})
dest = Path(__file__).resolve().parent / "olist_simple_orders.csv"
out.to_csv(dest, index=False)
# Join practice (code deck): the real payments for the same 1,000 orders (some orders have several payment rows),
# and a lookup of the 27 Brazilian states with their region.
pay = pd.read_csv(F / "olist_order_payments_dataset.csv")
pay = pay[pay["order_id"].isin(d.loc[out.index, "order_id"])].sort_values(["order_id", "payment_sequential"])
pay = pay.assign(order_id=pay["order_id"].str[:8])[["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"]]
pay.to_csv(dest.with_name("olist_simple_payments.csv"), index=False)
REG = {"North": "AC Acre,AP Amapa,AM Amazonas,PA Para,RO Rondonia,RR Roraima,TO Tocantins",
       "Northeast": "AL Alagoas,BA Bahia,CE Ceara,MA Maranhao,PB Paraiba,PE Pernambuco,PI Piaui,RN Rio Grande do Norte,SE Sergipe",
       "Centre-West": "DF Distrito Federal,GO Goias,MT Mato Grosso,MS Mato Grosso do Sul",
       "Southeast": "ES Espirito Santo,MG Minas Gerais,RJ Rio de Janeiro,SP Sao Paulo", "South": "PR Parana,RS Rio Grande do Sul,SC Santa Catarina"}
st = pd.DataFrame([(x[:2], x[3:], r) for r, v in REG.items() for x in v.split(",")], columns=["customer_state", "state_name", "region"])
st.sort_values("customer_state").to_csv(dest.with_name("brazil_states.csv"), index=False)
print("payments", pay.shape, "orders with >1 payment row", (pay.order_id.value_counts() > 1).sum(), "states", len(st))
print(dest.name, out.shape)
print(out.head(3).to_string())
print("missing", out.isna().sum()[out.isna().sum() > 0].to_dict(), "ids unique", out.order_id.is_unique)
late = out.days_late > 0; bad = out.review_score <= 2
print("mean delivery", round(out.delivery_days.mean(), 1), "top state", out.customer_state.value_counts(normalize=True).head(1).round(3).to_dict(),
      "late", round(late.mean(), 3), "bad if late", round(bad[late].mean(), 3), "bad on time", round(bad[~late].mean(), 3), "bad", round(bad.mean(), 3))
print("bad by items", bad.groupby(out.n_items.clip(upper=3)).mean().round(3).to_dict(), out.n_items.clip(upper=3).value_counts().to_dict())

# Cleaning lab: the RAW version of the same 1,000 orders, before anyone cleaned them, plus 60 real orders that were
# not delivered (or have no delivery date). Real problems: text timestamps, Portuguese categories (some missing,
# two with no English translation), orders marked delivered with no delivery date. Two problems are ADDED for practice
# and declared to students: 15 repeated rows, and 40 state codes typed differently (lower case, spaces).
import numpy as np
rng = np.random.default_rng(2026)
keep = set(d.loc[out.index, "order_id"])
first = i.sort_values("order_item_id").merge(p[["product_id", "product_category_name"]], on="product_id", how="left")
peritem = first.groupby("order_id").agg(n_items=("order_item_id", "size"), price=("price", "sum"), freight=("freight_value", "sum"),
                                        product_category=("product_category_name", "first")).reset_index()
other = o[~o["order_id"].isin(keep) & ((o["order_status"] != "delivered") | o["order_delivered_customer_date"].isna())]
other = other[other["order_id"].isin(peritem["order_id"])]
odd = other[other["order_status"] == "delivered"]                       # delivered, but no delivery date: all of them
other = pd.concat([odd, other[other["order_status"] != "delivered"].sample(60 - len(odd), random_state=2026)])
raw = pd.concat([o[o["order_id"].isin(keep)], other])
raw = raw.merge(c[["customer_id", "customer_city", "customer_state"]], on="customer_id").merge(peritem, on="order_id", how="left")
raw = raw.merge(r[["order_id", "review_score"]], on="order_id", how="left").sort_values("order_purchase_timestamp")
ts = lambda s: s.dt.strftime("%Y-%m-%d %H:%M:%S")
raw = pd.DataFrame({
    "order_id": raw["order_id"].str[:8], "order_status": raw["order_status"],
    "purchase_timestamp": ts(raw["order_purchase_timestamp"]), "delivered_timestamp": ts(raw["order_delivered_customer_date"]),
    "estimated_delivery": raw["order_estimated_delivery_date"].dt.strftime("%Y-%m-%d"),
    "customer_city": raw["customer_city"], "customer_state": raw["customer_state"], "product_category": raw["product_category"],
    "n_items": raw["n_items"], "price": raw["price"].round(2), "freight": raw["freight"].round(2), "review_score": raw["review_score"]}).reset_index(drop=True)
idx = rng.choice(len(raw), 40, replace=False)                          # added: state codes typed differently
fmt = [lambda s: s.lower(), lambda s: " " + s, lambda s: s + " ", lambda s: s.title()]
raw.loc[idx, "customer_state"] = [fmt[k % 4](raw.at[j, "customer_state"]) for k, j in enumerate(idx)]
dup = raw.iloc[rng.choice(len(raw), 15, replace=False)]                # added: repeated rows (an export run twice)
raw = pd.concat([raw, dup]).sort_values("purchase_timestamp", kind="stable").reset_index(drop=True)
raw.to_csv(dest.with_name("olist_raw_orders.csv"), index=False)
t.to_csv(dest.with_name("category_translation.csv"), index=False)
print("raw", raw.shape, raw["order_status"].value_counts().to_dict(), "dups", raw.duplicated().sum(),
      "missing", raw.isna().sum()[raw.isna().sum() > 0].to_dict())
