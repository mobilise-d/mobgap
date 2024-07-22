import pandas as pd
from pandas.testing import assert_frame_equal

from mobgap.initial_contacts import refine_gs


class TestRefineGs:
    def test_normal_case(self):
        input_ic_list = pd.DataFrame(
            {
                "ic": [5, 10, 20, 30, 40],
                "step_id": [0, 1, 2, 3, 4],
            }
        ).set_index("step_id")

        # The output gs-list is expected to be from the first IC to the last IC.
        # The output is_list is expected to be the input ic_list with the offset applied.
        refined_gs_list, refined_ic_list = refine_gs(input_ic_list)

        assert_frame_equal(
            refined_gs_list, pd.DataFrame({"start": [5], "end": [41], "r_gs_id": [0]}).set_index("r_gs_id")
        )
        assert_frame_equal(
            refined_ic_list,
            pd.DataFrame({"ic": [0, 5, 15, 25, 35], "step_id": [0, 1, 2, 3, 4], "r_gs_id": [0, 0, 0, 0, 0]}).set_index(
                ["r_gs_id", "step_id"]
            ),
        )
