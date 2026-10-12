import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_ingredient_stats(csv_path):
    df = pd.read_csv(csv_path)

    recipe_totals = df.groupby(['recipe_id', 'ingredient'])[['grams', 'pct_of_recipe']].sum().reset_index()
    stats = recipe_totals.groupby('ingredient')[['grams', 'pct_of_recipe']].agg(['mean', 'std', 'min', 'max'])
    stats['recipe_count'] = recipe_totals.groupby('ingredient').size()
    stats.columns = [f"{col[0]}_{col[1]}" for col in stats.columns]
    stats = stats.reset_index().round(2)

    stats.to_csv('ingredient_stats.csv', index=False)

    return stats


def get_category_stats(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Handle the trailing underscore on recipe_count if it exists
    if 'recipe_count_' in df.columns:
        df = df.rename(columns={'recipe_count_': 'recipe_count'})

    # Define appropriate aggregations for each column
    agg_funcs = {
        'grams_mean': 'mean',  # Average of the means across ingredients
        'grams_std': 'mean',  # Average of the standard deviations
        'grams_min': 'min',  # Absolute minimum across the category
        'grams_max': 'max',  # Absolute maximum across the category
        'pct_of_recipe_mean': 'mean',
        'pct_of_recipe_std': 'mean',
        'pct_of_recipe_min': 'min',
        'pct_of_recipe_max': 'max',
        'recipe_count': 'sum'  # Total instances of this category
    }

    # Group by category, aggregate, and round to 2 decimal places
    cat_stats = df.groupby('category').agg(agg_funcs).round(2).reset_index()

    # Sort by recipe_count to see the most dominant baking blocks first
    cat_stats = cat_stats.sort_values(by='recipe_count', ascending=False)

    cat_stats.to_csv(output_csv, index=False)
    return cat_stats


def generate_train_data(drop_na=True):
    df = pd.read_csv('Scraped Data/Clean ingredients/ingredients_metric.csv')

    if drop_na:
        # Find all recipe_ids that have at least one missing 'grams' value
        invalid_recipes = df[df['grams'].isna()]['recipe_id'].unique()
        # Filter the dataset to only keep valid recipes
        df = df[~df['recipe_id'].isin(invalid_recipes)]

    wide_df = df.pivot_table(
        index='recipe_id',
        columns='ingredient',
        values='grams',
        aggfunc='sum',
        fill_value=0
    ).add_prefix('grams_').reset_index()

    source_map = df[['recipe_id', 'source']].drop_duplicates()

    yt_metrics = pd.read_csv('Scraped Data/cookie-db/youtube_csv/recipes.csv')
    tt_metrics = pd.read_csv('Scraped Data/tiktok-cookies/CSV TikTok/recipes.csv')

    yt_metrics.rename(columns={'video_id': 'recipe_id'}, inplace=True)
    tt_metrics.rename(columns={'id': 'recipe_id'}, inplace=True)
    all_metrics = pd.concat([yt_metrics, tt_metrics], ignore_index=True)

    recipe_metrics = pd.merge(source_map, all_metrics, on='recipe_id', how='left')

    recipe_metrics['like_view_ratio'] = np.where(
        recipe_metrics['views'] > 0,
        recipe_metrics['likes'] / recipe_metrics['views'],
        0
    )

    final_df = pd.merge(
        wide_df,
        recipe_metrics[['recipe_id', 'likes', 'views', 'like_view_ratio']],
        on='recipe_id',
        how='left'
    ).dropna()

    drop_na = '_drop_na' if drop_na else ''
    final_df.to_csv(f'recipes_wide_format{drop_na}.csv', index=False)


def train_forest(drop_na=True):
    drop_na = '_drop_na' if drop_na else ''

    df = pd.read_csv(f'recipes_wide_format{drop_na}.csv')
    y = df['likes']
    X = df.drop(columns=['recipe_id', 'likes', 'views', 'like_view_ratio'], errors='ignore')

    # 80% Train 20% test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # HPO
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # 5-fold Cross Validation
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test MAE: {mean_absolute_error(y_test, best_model.predict(X_test)):.2f} likes")
    print(f"Test R²:  {r2_score(y_test, best_model.predict(X_test)):.4f}")


if __name__ == '__main__':
    get_ingredient_stats('Scraped Data/Clean ingredients/ingredients_metric.csv')
    # generate_train_data()
    # train_forest()

