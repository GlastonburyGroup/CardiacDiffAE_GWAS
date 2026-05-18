import argparse
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--res_path", help="path to the zip files", default=r"/project/ukbblatent/Out/Results/F20208_heart_1Ses_init_4ChTransfold0_prec32_pythaemodel-vae")
parser.add_argument("--res_name", help="path to the zip files", default=r"4ChTrans_VAE_latent10")
parser.add_argument("--out_path", help="path to the zip files", default=r"/group/glastonbury/soumick/LatentWorld/F20208/V1")

args = parser.parse_args()

source_file = f"{args.res_path}/Output/emb.h5"
destin_file = f"{args.out_path}/{args.res_name}.h5"

copyfile(source_file, destin_file)

print("Done!")