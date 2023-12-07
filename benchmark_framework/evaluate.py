import json
import argparse
import pandas as pd
import os.path as osp


def load_benchmark_data(data_path):
    metadata = json.load(open(data_path, 'r'))
    gt_data = {}
    for m in metadata:
        for i, f in enumerate(metadata[m]['files']):
            key = m + '/' + osp.splitext(osp.basename(f))[0]
            assert key not in gt_data
            if metadata[m]['label']['age'] != 'N/A':
                g = metadata[m]['label']['gender']
                gt_data[key] = {
                    'gt_skin': int(metadata[m]['label']['skin-type']),
                    'gt_gender': 'Other' if g == 'N/A' else g,
                    'gt_age': int(metadata[m]['label']['age']),
                    'gt_dark': 'Dark' if f in metadata[m]['dark_files'] else 'Bright'
                }

    return pd.DataFrame.from_dict(
        gt_data,
        orient='index',
        columns=['gt_skin', 'gt_gender', 'gt_age', 'gt_dark']
    ).rename_axis('id').reset_index()


def load_estimation_data(data_path):
    estimation_df = pd.read_csv(data_path, header=0, names=['path', 'est_age'])

    if 'CCMiniIMG' in data_path:
        estimation_df['id'] = estimation_df['path'].apply(lambda x: '/'.join(x.split('_raw')[0].split('/')[-2:]))
        estimation_df['frameNum'] = estimation_df['path'].apply(lambda x: int(osp.splitext(osp.basename(x))[0].split('_')[-1]))
    else:
        estimation_df['id'] = estimation_df['path'].apply(lambda x: '/'.join(x.split('/')[-3:-1]))
        estimation_df['frameNum'] = estimation_df['path'].apply(lambda x: int(osp.splitext(osp.basename(x))[0]))

    estimation_df = estimation_df.drop('path', axis=1)

    return estimation_df


def get_metrics_df(df):

    df['fame_lvl_abs_err'] = abs(df['gt_age'] - df['est_age'])

    metrics_df = (df.groupby(['id'], as_index=False).nth[0])[['id', 'gt_age', 'gt_gender', 'gt_skin', 'gt_dark']]
    metrics_df = metrics_df.set_index('id')
    metrics_df = metrics_df.join(df.groupby(['id'])['est_age'].median().rename('video_lvl_median_age'))
    metrics_df = metrics_df.join(df.groupby(['id'])['fame_lvl_abs_err'].mean().rename('video_lvl_mae'))
    metrics_df = metrics_df.join(df.groupby(['id'])['fame_lvl_abs_err'].std().rename('video_lvl_std'))
    metrics_df['video_lvl_abs_err'] = (metrics_df['video_lvl_median_age'] - metrics_df['gt_age']).abs()

    return metrics_df


def get_protocol_metrics(df):

    metrics = {
        'Offline': {
            'MAE': {'Overall': df['video_lvl_abs_err'].mean()}
        },
        'Online': {
            'tMAE': {'Overall': df['video_lvl_mae'].mean()},
            'tSTD': {'Overall': df['video_lvl_std'].mean()}
        }
    }

    for gender in ['Female', 'Male', 'Other']:
        sub_df = df[df['gt_gender'] == gender]
        metrics['Offline']['MAE'][gender] = sub_df['video_lvl_abs_err'].mean()
        metrics['Online']['tMAE'][gender] = sub_df['video_lvl_mae'].mean()
        metrics['Online']['tSTD'][gender] = sub_df['video_lvl_std'].mean()

    for skin in [1, 2, 3, 4, 5, 6]:
        sub_df = df[df['gt_skin'] == skin]
        metrics['Offline']['MAE']['SkinType' + str(skin)] = sub_df['video_lvl_abs_err'].mean()
        metrics['Online']['tMAE']['SkinType' + str(skin)] = sub_df['video_lvl_mae'].mean()
        metrics['Online']['tSTD']['SkinType' + str(skin)] = sub_df['video_lvl_std'].mean()

    for dark in ['Bright', 'Dark']:
        sub_df = df[df['gt_dark'] == dark]
        metrics['Offline']['MAE'][dark] = sub_df['video_lvl_abs_err'].mean()
        metrics['Online']['tMAE'][dark] = sub_df['video_lvl_mae'].mean()
        metrics['Online']['tSTD'][dark] = sub_df['video_lvl_std'].mean()

    return metrics


def main(args):

    print('Loading benchmark data..')
    meta_df = load_benchmark_data(args.metadata_path)

    print('Loading estimation data..')
    estimation_df = load_estimation_data(args.estimation_data_path)

    print('Merging benchmark and estimation data..')
    merged_df = estimation_df.merge(meta_df, on='id')

    print('Calculating protocol metrics..')
    metrics_df = get_metrics_df(merged_df)
    metrics = get_protocol_metrics(metrics_df)
    print(json.dumps(metrics, indent=2))

    print(f'Saving results to {args.output_results_path}')
    with open(args.output_results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Evaluation finished!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', default='benchmark_data/CCMiniVID/CCMiniVID-A_metadata.json', type=str)
    parser.add_argument('--estimation_data_path', default='evaluation_data/CCMiniVID-A_TCN4_estimations.csv', type=str)
    parser.add_argument('--output_results_path', default='evaluation_data/CCMiniVID-A_TCN4_metrics.json', type=str)
    args = parser.parse_args()

    main(args)
