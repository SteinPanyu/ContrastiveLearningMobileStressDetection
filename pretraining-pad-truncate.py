from pathlib import Path
import pandas as pd

def filter_missing_sequences(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    common_timestamps = set.intersection(*[set(df['timestamp'].unique().tolist()) for df in dfs])
    return [df.loc[df['timestamp'].isin(common_timestamps)] for df in dfs]

def pad_all_sensors(sensors : list[pd.DataFrame], desired_length=60) -> list[pd.DataFrame]:

    def pad_and_truncate_dataframe_sequences(sequences: pd.DataFrame,  desired_length=desired_length) -> pd.DataFrame:
        
        def pad_and_truncate_individual_sequence(sequence: pd.DataFrame, desired_length=desired_length) -> pd.DataFrame:      
            current_length = len(sequence)
            
            if current_length==desired_length:
                return sequence
            
            if current_length < desired_length:
                last_value = sequence.iloc[-1]
                padding = [last_value] * (desired_length - current_length)
            
                padded_sequence = pd.concat([sequence, pd.DataFrame(padding)],)
                return padded_sequence
            
            return sequence.iloc[:desired_length]
        
        
        return pd.concat([pad_and_truncate_individual_sequence(sequence=sequence,  desired_length=desired_length) for _,sequence in sequences.groupby('timestamp')]).set_index('timestamp')
        
    return [pad_and_truncate_dataframe_sequences(sensor) for sensor in sensors]

subjects = [f'P{i:02}' for i in range(1,81) if i not in (22,27,59,65)]
unlabeled_paths = dict([(subject, list(Path('Intermediate/proc/').glob(f'{subject}*unlabeled.csv'))) for subject in subjects])
for particpant, paths in unlabeled_paths.items():
    sensors = [pd.read_csv(p, usecols=lambda c: c not in ('label', 'pcode')) for p in paths]
    sensors = filter_missing_sequences(sensors)
    sensors = pad_all_sensors(sensors)
    pd.concat(sensors, axis=1).to_csv(Path(f'Intermediate/proc/unlabeled_joined/{particpant}_unlabeled.csv'))
    print(f"Finished writing unlabeled sequences for participant {particpant}")
