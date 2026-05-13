import pandas as pd

df = pd.read_csv("output/preprocessed/messages_community.csv")
df["PostDate"] = pd.to_datetime(df["PostDate"], errors="coerce")
df = df.sort_values(["ForumTopicID", "PostDate"]).reset_index(drop=True)
df["is_initial_post"] = df.groupby("ForumTopicID").cumcount() == 0
df["wc"] = df["MessageText"].fillna("").apply(lambda x: len(x.split()))

short = df[df["wc"] < 3].copy()

print("\n=== SHORT INITIAL POSTS (< 3 words) ===")
short_initial = short[short["is_initial_post"]][["ForumTopicID", "MessageText", "wc"]].sort_values("wc")
print(short_initial.to_string())

print("\n=== SHORT REPLIES (< 3 words) — first 30 ===")
short_replies = short[~short["is_initial_post"]][["ForumTopicID", "MessageText", "wc"]].sort_values("wc")
print(short_replies.head(30).to_string())