#USAGE python UmbSam_EQUILIBRATION.py $INPUT_DIR $CENTROID_A_NAME $CENTROID_B_NAME $OUTPUT_DIR
import sys
from FultonMarket.Unilateral_Trailblazing import Unilateral_Umbrella_Trailblazer

assert len(sys.argv) == 5, "You may have the wrong number of arguments :("
args = dict(input_dir = sys.argv[1],
            centroid_A_name = sys.argv[2],
            centroid_B_name = sys.argv[3],
            save_dir = sys.argv[4])


cb2_intracellular_inds = np.array([49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311])
intra_inds_dict = {}

for leg_iden in ["A", "B"]:
    if args[f"centroid_{leg_iden}_name"] in ["centroid_6", "centroid_13"]:
        #subtract 22 from these indices
        intra_inds_dict[leg_iden] = cb2_intracellular_inds - 22
    else:
        #subtract 23
        intra_inds_dict[leg_iden] = cb2_intracellular_inds - 23


#Run Block
UT = Unilateral_Umbrella_Trailblazer(**args)

UT.assign_spring_attributes(intracellular_residue_indsA=intra_inds_dict["A"],
                            intracellular_residue_indsB=intra_inds_dict["B"])

UT.run_trailblazing()

UT.save_results()
