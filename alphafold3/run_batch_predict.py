import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from absl import app, flags
from pathlib import Path
from alphafold3.utils.predict_utils import (
    make_model_runner,
    load_base_input,
    make_base_input,
    get_af3_prediction,
    convert_cif_to_pdb,
)


def main(_):
    target_path = Path("/home/fuxin/lab/wwt/predict/b2m_0123")
    task_tag = "b2m"

    receptor_data_path = Path(
        "/home/fuxin/lab/wwt/predict/b2m_cyclic/compete_b2m/b2m_115_data.json"
    )
    if receptor_data_path.exists():
        receptor_input = load_base_input(receptor_data_path)
    else:
        receptor_seq = [
            "IIGGEFTTIENQPWFAAIYRRHRGGSVTYVCGGSLISPCWVISATHCFIDYPKKEDYIVYLGRSRLNSNTQGEMKFEVENLILHKDYSADTLAHHNDIALLKIRSKEGRCAQPSRTIQTIALPSMYNDPQFGTSCEITGFGKEQSTDYLYPEQLKMTVVKLISHRECQQPHYYGSEVTTKMLCAADPQWKTDSCQGDSGGPLVCSLQGRMTLTGIVSWGRGCALKDKPGVYTRVSHFLPWIRSHT"
        ]
        receptor_input = make_base_input(
            task_tag, receptor_seq, [1], True, receptor_data_path.parent
        )

    model_runner = make_model_runner(
        device_index=0,
        model_dir="/data/soft/AF3-Model/",
        num_samples=5,
        use_bonds=True,
        use_offsets=False,
    )

    predict_sequences = [
        "CVRPGTHGNSC",
        "CYRPGTHGNSC",
        "CVRPGTHGDSC",
        "CVRPGTHGKSC",
        "CFRPGTHGNSC",
        "CVRPGFHGNSC",
    ]
    ids = [
        "manual_1",
        "manual_2",
        "manual_3",
        "manual_4",
        "manual_5",
        "manual_6",
    ]
    bonds = [((0, "SG"), (10, "SG"))]
    for i, (sequence, id) in enumerate(zip(predict_sequences, ids)):
        receptor_input.name = id
        cif_file = f"{str(target_path)}/{id}/{id.lower()}_model.cif"
        if not Path(cif_file).exists():

            peptide_chain_id = get_af3_prediction(
                receptor_input,
                sequence,
                model_runner,
                target_path / f"{id}",
                bonds=bonds,
                modifications=[],
                ligand_bonds=[],
            )


if __name__ == "__main__":
    app.run(main)
