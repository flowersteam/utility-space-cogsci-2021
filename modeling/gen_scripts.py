from pandas import read_csv
import argparse


def main(data_path, save_to, name, lim):
    nfits, save_as, test = 1, 'results', '1'
    sids = read_csv(data_path).loc[:, 'sid'].unique()
    head = \
'''#!/bin/sh
#SBATCH --mincpus 24
#SBATCH -C bora
#SBATCH -p routage
#SBATCH -t 6:00:00
#SBATCH -e {0}/logs/run{1}.err
#SBATCH -o {0}/logs/run{1}.out
'''
    stride = 24
    batches = []
    for i, j in enumerate(range(0, len(sids), stride)):
        if lim and (i+1 > lim):
            break
        jobs = '\n'.join([f'python fit_models.py --sid {sid} --nfits {nfits} --save_to {save_as} --test {test} &' for sid in sids[j:min(j+stride, len(sids))]])
        filename = name+f'{i}'.zfill(2)
        script = head.format(save_as, i) + jobs + '\nwait'
        with open(save_to+'/'+filename+'.sh', 'w') as script_file:
            script_file.write(script)
        batches.append(filename+'.sh')

    with open(save_to+'/'+'runall.sh', 'w') as runall_script:
        runall_script.write(
            '#!/bin/sh\n' + '\n'.join([f'sbatch {batch}' for batch in batches])
        )

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='path to data containing subject IDs', type=str)
parser.add_argument('--loc', help='relative path to save scripts', type=str)
parser.add_argument('--name', help='name given to each batch script', type=str)
parser.add_argument('--lim', help='maximum number of batches to generate', type=int)
args = parser.parse_args()

lim = args.lim if args.lim else None

main(args.data, args.loc, args.name, args.lim)
