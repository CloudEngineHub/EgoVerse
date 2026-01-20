from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Boolean,
    Float,
    text,
)
from egomimic.utils.aws.aws_sql import (
    TableRow,
    add_episode,
    update_episode,
    create_default_engine,
    episode_hash_to_table_row,
    delete_episodes,
    episode_table_to_df,
    delete_all_episodes,
)
import pandas as pd

engine = create_default_engine()

df = episode_table_to_df(engine)

# Filter for task "fold clothes"
df_fold = df[df['task'].isin(['fold_clothes', 'fold clothes'])].copy()

# Convert scene to string for consistent comparison
df_fold['scene'] = df_fold['scene'].astype(str)

# Expected denominators (C) from image data
# Format: (operator_name, scene): denominator
expected_denominators = {
    ('Aniketh', '1'): 32, ('Aniketh', '2'): 1, ('Aniketh', '3'): 1, ('Aniketh', '4'): 1,
    ('Aniketh', '5'): 1, ('Aniketh', '6'): 1, ('Aniketh', '7'): 1, ('Aniketh', '8'): 1,
    ('Jenny', '1'): 16, ('Jenny', '2'): 1, ('Jenny', '3'): 1, ('Jenny', '4'): 1,
    ('Jenny', '5'): 1, ('Jenny', '6'): 1, ('Jenny', '7'): 1, ('Jenny', '8'): 1,
    ('Baoyu', '1'): 8, ('Baoyu', '2'): 1, ('Baoyu', '3'): 1, ('Baoyu', '4'): 1,
    ('Baoyu', '5'): 1, ('Baoyu', '6'): 1, ('Baoyu', '7'): 1, ('Baoyu', '8'): 1,
    ('Lawrence', '1'): 8, ('Lawrence', '2'): 1, ('Lawrence', '3'): 1, ('Lawrence', '4'): 1,
    ('Lawrence', '5'): 1, ('Lawrence', '6'): 1, ('Lawrence', '7'): 1, ('Lawrence', '8'): 1,
    ('Ryan', '1'): 4, ('Ryan', '2'): 4, ('Ryan', '3'): 4, ('Ryan', '4'): 4,
    ('Ryan', '5'): 4, ('Ryan', '6'): 4, ('Ryan', '7'): 4, ('Ryan', '8'): 4,
    ('Pranav', '1'): 4, ('Pranav', '2'): 4, ('Pranav', '3'): 4, ('Pranav', '4'): 4,
    ('Pranav', '5'): 4, ('Pranav', '6'): 4, ('Pranav', '7'): 4, ('Pranav', '8'): 4,
    ('Nadun', '1'): 4, ('Nadun', '2'): 4, ('Nadun', '3'): 4, ('Nadun', '4'): 4,
    ('Nadun', '5'): 4, ('Nadun', '6'): 4, ('Nadun', '7'): 4, ('Nadun', '8'): 4,
    ('Yangcen', '1'): 4, ('Yangcen', '2'): 4, ('Yangcen', '3'): 4, ('Yangcen', '4'): 4,
    ('Yangcen', '5'): 4, ('Yangcen', '6'): 4, ('Yangcen', '7'): 4, ('Yangcen', '8'): 4,
    ('Zhenyang', '1'): 2, ('Zhenyang', '2'): 2, ('Zhenyang', '3'): 2, ('Zhenyang', '4'): 2,
    ('Zhenyang', '5'): 2, ('Zhenyang', '6'): 2, ('Zhenyang', '7'): 2, ('Zhenyang', '8'): 2,
    ('Woolchul', '1'): 2, ('Woolchul', '2'): 2, ('Woolchul', '3'): 2, ('Woolchul', '4'): 2,
    ('Woolchul', '5'): 2, ('Woolchul', '6'): 2, ('Woolchul', '7'): 2, ('Woolchul', '8'): 2,
    ('Shuo', '1'): 2, ('Shuo', '2'): 2, ('Shuo', '3'): 2, ('Shuo', '4'): 2,
    ('Shuo', '5'): 2, ('Shuo', '6'): 2, ('Shuo', '7'): 2, ('Shuo', '8'): 2,
    ('Liqian', '1'): 2, ('Liqian', '2'): 2, ('Liqian', '3'): 2, ('Liqian', '4'): 2,
    ('Liqian', '5'): 2, ('Liqian', '6'): 2, ('Liqian', '7'): 2, ('Liqian', '8'): 2,
    ('Xinchen', '1'): 2, ('Xinchen', '2'): 2, ('Xinchen', '3'): 2, ('Xinchen', '4'): 2,
    ('Xinchen', '5'): 2, ('Xinchen', '6'): 2, ('Xinchen', '7'): 2, ('Xinchen', '8'): 2,
    ('Rohan', '1'): 2, ('Rohan', '2'): 2, ('Rohan', '3'): 2, ('Rohan', '4'): 2,
    ('Rohan', '5'): 2, ('Rohan', '6'): 2, ('Rohan', '7'): 2, ('Rohan', '8'): 2,
    ('David', '1'): 2, ('David', '2'): 2, ('David', '3'): 2, ('David', '4'): 2,
    ('David', '5'): 2, ('David', '6'): 2, ('David', '7'): 2, ('David', '8'): 2,
    ('Vaibhav', '1'): 2, ('Vaibhav', '2'): 2, ('Vaibhav', '3'): 2, ('Vaibhav', '4'): 2,
    ('Vaibhav', '5'): 2, ('Vaibhav', '6'): 2, ('Vaibhav', '7'): 2, ('Vaibhav', '8'): 2,
    ('Mengying', '1'): 2, ('Mengying', '2'): 2, ('Mengying', '3'): 2, ('Mengying', '4'): 2,
    ('Mengying', '5'): 2, ('Mengying', '6'): 2, ('Mengying', '7'): 2, ('Mengying', '8'): 2,
}

# Operator mapping to Operator N format
operator_mapping = {
    'Aniketh': 'Operator 1 (Aniketh)',
    'Jenny': 'Operator 2 (Jenny)',
    'Baoyu': 'Operator 3 (Baoyu)',
    'Lawrence': 'Operator 4 (Lawrence)',
    'Ryan': 'Operator 5 (Ryan)',
    'Pranav': 'Operator 6 (Pranav)',
    'Nadun': 'Operator 7 (Nadun)',
    'Yangcen': 'Operator 8 (Yangcen)',
    'Zhenyang': 'Operator 9 (Zhenyang)',
    'Woolchul': 'Operator 10 (Woolchul)',
    'Shuo': 'Operator 11 (Shuo)',
    'Liqian': 'Operator 12 (Liqian)',
    'Xinchen': 'Operator 13 (Xinchen)',
    'Rohan': 'Operator 14 (Rohan)',
    'David': 'Operator 15 (David)',
    'Vaibhav': 'Operator 16 (Vaibhav)',
    'Mengying': 'Operator 17 (Mengying)',
}

# Check if processed_path is not empty (not null and not empty string)
df_fold['has_processed_path'] = (
    df_fold['processed_path'].notna() & 
    (df_fold['processed_path'].astype(str).str.strip() != '')
)

# Calculate metrics for each operator-scene combination
results = []
for operator in operator_mapping.keys():
    for scene in ['1', '2', '3', '4', '5', '6', '7', '8']:
        # Filter for this operator-scene combination
        op_scene_df = df_fold[
            (df_fold['operator'] == operator) & 
            (df_fold['scene'] == scene)
        ]
        
        # A: Count where processed_path is not empty
        A = op_scene_df['has_processed_path'].sum()
        
        # B: Total count for operator-scene
        B = len(op_scene_df)
        
        # C: Expected denominator from image data
        C = expected_denominators.get((operator, scene), 0)
        
        results.append({
            'operator': operator_mapping[operator],
            'scene': scene,
            'A': A,
            'B': B,
            'C': C,
        })

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Create formatted output DataFrame directly from results
scenes = ['1', '2', '3', '4', '5', '6', '7', '8']
output_data = {}

for operator_display in operator_mapping.values():
    output_data[operator_display] = {}
    for scene in scenes:
        # Find the result for this operator-scene combination
        result = results_df[
            (results_df['operator'] == operator_display) & 
            (results_df['scene'] == scene)
        ]
        
        if len(result) > 0:
            A_val = int(result['A'].iloc[0])
            B_val = int(result['B'].iloc[0])
            C_val = int(result['C'].iloc[0])
        else:
            # Get operator name from display name
            operator_name = [k for k, v in operator_mapping.items() if v == operator_display][0]
            A_val = 0
            B_val = 0
            C_val = expected_denominators.get((operator_name, scene), 0)
        
        output_data[operator_display][f'Scenario {scene}'] = f"{A_val} / {B_val} / {C_val}"

# Create final output DataFrame
output_df = pd.DataFrame(output_data).T
output_df = output_df.reindex(columns=[f'Scenario {s}' for s in scenes])
# Ensure operators are in the correct order
operator_order = [operator_mapping[op] for op in operator_mapping.keys()]
output_df = output_df.reindex(operator_order)

# Save to CSV with proper formatting
# Add index name for the operator column
output_df.index.name = 'Time (minutes)'
output_df.to_csv('operator_scene_statistics.csv')
print("CSV file saved as 'operator_scene_statistics.csv'")
print(f"\nTotal operators: {len(output_df)}")
print(f"Total scenarios: {len(output_df.columns)}")
print("\nFirst few rows:")
print(output_df.head())
print("\nLast few rows:")
print(output_df.tail())