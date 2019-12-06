from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
sys.path.append('/afs/cern.ch/work/e/efol/public/Beta-Beat.src/')
from Utilities import tfs_pandas
from madx import madx_wrapper
from multiprocessing import Pool
import random
from collections import OrderedDict

MY_DIR = '/afs/cern.ch/work/e/efol/public/ML/LHC_quad_prediction/40cm_collision_realisticTriplet/40cm_2Beams/simulations'
NUM_SIM = 8
MADX_PATH = "/afs/cern.ch/user/m/mad/madx/releases/last-rel/madx-linux64-intel"
COLL_40CM = '/afs/cern.ch/eng/lhc/optics/runII/2018/PROTON/opticsfile.19'
MODEL_B1 = os.path.join(MY_DIR, "twiss.b1nominal.dat")
MODEL_B2 = os.path.join(MY_DIR, 'twiss.b2nominal.dat')
NOMINAL_TWISS_TEMPL = os.path.join(MY_DIR, 'job2018.nominal.madx')
MAGNETS_TEMPLATE_B1 = os.path.join(MY_DIR, 'job.magneterrors_b1.madx')
MAGNETS_TEMPLATE_B2 = os.path.join(MY_DIR, 'job.magneterrors_b2.madx')
MADX_OUTPUT = os.path.join(MY_DIR, 'training')
PATH_SAVED_DATA = os.path.join(MY_DIR, 'training_data.npy')


def create_samples_with_validation(index):
    sample = None
    print("Start creating dataset")
    np.random.seed(seed=None)
    print("Doing index: ", str(index))
    seed = random.randint(0, 999999999)
    with open(MAGNETS_TEMPLATE_B1, 'r') as template:
        template_str = template.read()
    madx_wrapper.resolve_and_run_string(template_str % {"DIR": MADX_OUTPUT, "INDEX": str(index), "OPTICS": COLL_40CM, "SEED": seed}, madx_path=MADX_PATH)
    with open(MAGNETS_TEMPLATE_B2, 'r') as template:
        template_str = template.read()
    madx_wrapper.resolve_and_run_string(template_str % {"DIR": MADX_OUTPUT, "INDEX": str(index), "OPTICS": COLL_40CM, "SEED": seed}, madx_path=MADX_PATH)
    # save data in sample
    b1_errors_file_path = os.path.join(MADX_OUTPUT, "b1_errors_{}.tfs".format(index))
    b2_errors_file_path = os.path.join(MADX_OUTPUT, "b2_errors_{}.tfs".format(index))
    b1_tw_perturbed_path = os.path.join(MADX_OUTPUT, "b1_twiss_{}.tfs".format(index))
    b2_tw_perturbed_path = os.path.join(MADX_OUTPUT, "b2_twiss_{}.tfs".format(index))

    b1_tw_before_match = os.path.join(MADX_OUTPUT, "b1_twiss_before_match_{}.tfs".format(index))
    b1_tw_after_match = os.path.join(MADX_OUTPUT, "b1_twiss_after_match_{}.tfs".format(index))

    b2_tw_before_match = os.path.join(MADX_OUTPUT, "b2_twiss_before_match_{}.tfs".format(index))
    b2_tw_after_match = os.path.join(MADX_OUTPUT, "b2_twiss_after_match_{}.tfs".format(index))

    common_errors_path = os.path.join(MADX_OUTPUT, "common_errors_{}.tfs".format(index))
    if os.path.isfile(b1_tw_perturbed_path) and os.path.isfile(b2_tw_perturbed_path):
        beta_star_b1, delta_mux_b1, delta_muy_b1, delta_dx_b1 = get_input_for_beam(b1_tw_perturbed_path, MODEL_B1, 1)
        beta_star_b2, delta_mux_b2, delta_muy_b2, delta_dx_b2 = get_input_for_beam(b2_tw_perturbed_path, MODEL_B2, 2)
        errors = get_errors_from_file(common_errors_path, b1_errors_file_path, b2_errors_file_path, b1_tw_before_match, b1_tw_after_match, b2_tw_before_match, b2_tw_after_match)
        
        # TODO: when creating validation set: activate it
        # add_mqts_to_errtab(b1_errors_file_path, b1_tw_before_match, b1_tw_after_match)
        # add_mqts_to_errtab(b2_errors_file_path, b2_tw_before_match, b2_tw_after_match)
        
        sample = beta_star_b1, beta_star_b2, delta_mux_b1, delta_muy_b1, delta_dx_b1, delta_mux_b2, delta_muy_b2, delta_dx_b2, np.array(errors, dtype=float)

        os.remove(b1_tw_perturbed_path)
        os.remove(b2_tw_perturbed_path)
        os.remove(b1_errors_file_path)
        os.remove(b2_errors_file_path)

        os.remove(b1_tw_before_match)
        os.remove(b1_tw_after_match)
        os.remove(b2_tw_before_match)
        os.remove(b2_tw_after_match)

        os.remove(common_errors_path)
    return sample


def add_mqts_to_errtab(errtab_path, unmatched_path, matched_path):
    errtab_tfs = tfs_pandas.read_tfs(errtab_path).set_index("NAME", drop=False)
    unmatched_tfs = tfs_pandas.read_tfs(unmatched_path).set_index("NAME")
    matched_tfs = tfs_pandas.read_tfs(matched_path).set_index("NAME")
    mqt_names = [name for name in unmatched_tfs.index.values if "MQT." in name]
    mqt_changes = np.array(matched_tfs.loc[mqt_names, "K1L"].values - unmatched_tfs.loc[mqt_names, "K1L"].values, dtype=float).round(decimals=10)
    errtab_tfs.loc[mqt_names, "K1L"] = mqt_changes
    # TODO: when creating validation set: replace original errtab file with new one - write the file!
    tfs_pandas.write_tfs(errtab_path, errtab_tfs, save_index=True)


# Read error table (as tfs), return k1l absolute
def get_errors_from_file(common_errors_path, b1_path, b2_path, b1_tw_before_match, b1_tw_after_match, b2_tw_before_match, b2_tw_after_match):
    #all errors in form: all beam 1 quads, all beam 2 quads (so triplets are repeated) -> due to distributions of bpms in input data
    all_errors = []
    triplet_errors_tfs = tfs_pandas.read_tfs(common_errors_path).set_index("NAME")
    triplet_errors = triplet_errors_tfs.K1L.values
    # K1L != 0 -> not mqts, save
    tfs_error_file_b1 = tfs_pandas.read_tfs(b1_path).set_index("NAME")
    # replace K1L of MQT in original table (0) with matched - unmatched difference, per knob (2 different values for all MQTs)
    b1_unmatched = tfs_pandas.read_tfs(b1_tw_before_match).set_index("NAME")
    b1_matched = tfs_pandas.read_tfs(b1_tw_after_match).set_index("NAME")
    mqt_names_b1 = [name for name in b1_unmatched.index.values if "MQT." in name]
    mqt_changes = np.array(b1_matched.loc[mqt_names_b1, "K1L"].values - b1_unmatched.loc[mqt_names_b1, "K1L"].values, dtype=float).round(decimals=10)
    mqt_errors_b1 = np.unique(mqt_changes).round(decimals=10)
    mqt_errors_b1 = [k for k in mqt_errors_b1 if k != 0]

    other_magnets_names_b1 = [name for name in tfs_error_file_b1.index.values if ("MQT." not in name and "MQX" not in name)]
    other_errors_b1 = tfs_error_file_b1.loc[other_magnets_names_b1, "K1L"].values
    tfs_error_file_b2 = tfs_pandas.read_tfs(b2_path).set_index("NAME")
    # replace K1L of MQT in original table (0) with matched - unmatched difference, per knob (2 different values for all MQTs)
    b2_unmatched = tfs_pandas.read_tfs(b2_tw_before_match).set_index("NAME")
    b2_matched = tfs_pandas.read_tfs(b2_tw_after_match).set_index("NAME")
    mqt_names_b2 = [name for name in b2_unmatched.index.values if "MQT." in name]
    mqt_changes_b2 = np.array(b2_matched.loc[mqt_names_b2, "K1L"].values - b2_unmatched.loc[mqt_names_b2, "K1L"].values, dtype=float).round(decimals=10)
    mqt_errors_b2 = np.unique(mqt_changes_b2).round(decimals=10)
    mqt_errors_b2 = [k for k in mqt_errors_b2 if k != 0]

    other_magnets_names_b2 = [name for name in tfs_error_file_b2.index.values if ("MQT." not in name and "MQX" not in name)]
    other_errors_b2 = tfs_error_file_b2.loc[other_magnets_names_b2, "K1L"].values

    all_errors.extend(triplet_errors)
    all_errors.extend(other_errors_b1)
    all_errors.extend(mqt_errors_b1)
    all_errors.extend(other_errors_b2)
    all_errors.extend(mqt_errors_b2)

    return all_errors


def create_nominal_twiss(optics):
    with open(NOMINAL_TWISS_TEMPL, 'r') as template:
        template_str = template.read()
    madx_wrapper.resolve_and_run_string(template_str % {"OPTICS": COLL_40CM}, madx_path=MADX_PATH)


def get_input_for_beam(tw_perturbed_path, mdl_path, beam):
    ip_bpms_b1 = ["BPMSW.1L1.B1", "BPMSW.1R1.B1", "BPMSW.1L2.B1", "BPMSW.1R2.B1", "BPMSW.1L5.B1", "BPMSW.1R5.B1", "BPMSW.1L8.B1", "BPMSW.1R8.B1"]
    ip_bpms_b2 = ["BPMSW.1L1.B2", "BPMSW.1R1.B2", "BPMSW.1L2.B2", "BPMSW.1R2.B2", "BPMSW.1L5.B2", "BPMSW.1R5.B2", "BPMSW.1L8.B2", "BPMSW.1R8.B2"]
    tw_perturbed = tfs_pandas.read_tfs(tw_perturbed_path).set_index("NAME")
    mdl = tfs_pandas.read_tfs(mdl_path).set_index("NAME")
    tw_perturbed_reindexed = tw_perturbed.loc[mdl.index, :]
    beta_star = []
    ip_bpms = ip_bpms_b1 if beam == 1 else ip_bpms_b2
    for bpm in ip_bpms:
        beta_star.append(tw_perturbed_reindexed.loc[bpm, "BETX"] - mdl.loc[bpm, "BETX"])
        beta_star.append(tw_perturbed_reindexed.loc[bpm, "BETY"] - mdl.loc[bpm, "BETY"])
    delta_mux = tw_perturbed_reindexed.MUX - mdl.MUX
    delta_muy = tw_perturbed_reindexed.MUY - mdl.MUY
    delta_dx = tw_perturbed_reindexed.NDX - mdl.NDX
    return beta_star, delta_mux, delta_muy, delta_dx


def main():
    # create_nominal_twiss(COLL_40CM)
    # create_samples_with_validation("test")
    print("Start")
    all_samples = []
    pool = Pool(processes=8)
    all_samples = pool.map(create_samples_with_validation, range(NUM_SIM))
    pool.close()
    pool.join()
    np.save(PATH_SAVED_DATA, np.array(all_samples, dtype=object))
    all_samples = np.load(PATH_SAVED_DATA)
    real_samples = []
    for sample in all_samples:
        if sample is not None:
            real_samples.append(sample)
    print("{} samples created.".format(len(real_samples)))


if __name__ == "__main__":
    main()