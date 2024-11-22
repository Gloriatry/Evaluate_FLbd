import pandas as pd

input_file = '../project/loan.csv'
df = pd.read_csv(input_file)

unique_states = df['addr_state'].unique()

for state in unique_states:
    state_df = df[df['addr_state'] == state]
    
    output_file = f'../project/data/loan/loan_{state}.csv'
    
    state_df.to_csv(output_file, index=False)
    
    print(f'Data for {state} has been written to {output_file}')