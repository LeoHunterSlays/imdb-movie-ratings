from functools import partial
from typing import Dict, Iterable, Optional, Text

import pandas as pd

read_csv = partial(pd.read_csv, compression="gzip", sep="\t", low_memory=False)
clean_str_return_int = lambda x: int(x) if x != "\\N" else -1
clean_str_return_none = lambda x: x if x != "\\N" else None


def parse_basics(
    read_path: Text,
    title_type: Optional[Text] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Read in title basics info, clean and filter"""
    df = read_csv(read_path)

    # Filter by `titleType`
    if title_type is not None:
        df = df[df["titleType"] == title_type]

    # Use data where `runtimeMinutes` is available
    df["year"] = df["startYear"].apply(clean_str_return_int)
    df["runtimeMinutes"] = df["runtimeMinutes"].apply(clean_str_return_int)
    df = df[df["runtimeMinutes"] > 0]

    # Filter data by year
    df["year"] = df["startYear"].apply(clean_str_return_int)
    if start_year is not None:
        df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]

    features_to_use = ["year", "runtimeMinutes", "genres"]

    return df[["tconst"] + features_to_use]


def parse_akas(
    read_path: Text,
    titles_to_keep: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Read in title akas info, clean and filter"""
    df = read_csv(read_path)

    if titles_to_keep is not None:
        df = df[df["titleId"].isin(titles_to_keep)].rename(
            columns={"titleId": "tconst"}
        )

    # Compute localization score using `ordering`
    agg_df = (
        df.groupby("tconst", as_index=False)
        .agg({"ordering": max})
        .rename(columns={"ordering": "local_score"})
    )

    return agg_df


def parse_principals(
    read_path: Text,
    titles_to_keep: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Read in title principals info, clean, filter and aggregate"""
    df = read_csv(read_path)

    if titles_to_keep is not None:
        df = df[df["tconst"].isin(titles_to_keep)]

    agg_df = (
        df.groupby("tconst", as_index=False)
        .agg({"category": "nunique"})
        .rename(columns={"category": "n_uniq_cast"})
    )

    return agg_df


def parse_ratings(
    read_path: Text,
    titles_to_keep: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Read in title ratings info, clean and filter"""
    df = read_csv(read_path)

    if titles_to_keep is not None:
        df = df[df["tconst"].isin(titles_to_keep)]

    features_to_use = ["averageRating", "numVotes"]

    return df[["tconst"] + features_to_use]


def parse_and_join_data(
    basic_kwargs: Dict,
    aka_kwargs: Dict,
    principal_kwargs: Dict,
    rating_kwargs: Dict,
) -> pd.DataFrame:
    """Read cleaned datasets and join them"""
    basics = parse_basics(**basic_kwargs)
    titles_to_keep = set(basics["tconst"])

    aka_kwargs["titles_to_keep"] = titles_to_keep
    principal_kwargs["titles_to_keep"] = titles_to_keep
    rating_kwargs["titles_to_keep"] = titles_to_keep

    akas = parse_akas(**aka_kwargs)
    principals = parse_principals(**principal_kwargs)
    ratings = parse_ratings(**rating_kwargs)

    joined_df = (
        basics.merge(akas, on="tconst")
        .merge(principals, on="tconst")
        .merge(ratings, on="tconst")
    )

    return joined_df
