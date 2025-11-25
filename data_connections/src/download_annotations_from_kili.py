import argparse
import json
import logging
import os
import time
from pathlib import Path

from kili.client import Kili
from models import SiltDetection, SiltFile, SiltObject, SiltSensor

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to capture (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # Define the timestamp format
)

# Get a logger for your module (or use the root logger directly)
logger = logging.getLogger(__name__)

# Replace with your Kili API key and project ID
KILI_API_KEY = os.environ.get("KILI_API_KEY")


def pull_new_annotations_for_project(
    project_id: str, asset_id_list: list[str] = None
) -> tuple[list[str], list[SiltDetection]]:
    """Function to download annotations from Kili for a given project.

    Args:
        project_id (str): id string for the project in question (note: not the common name, the string in the Kili URL)
        asset_id_list (list[str]): list of asset IDs that have already been pulled, so we don't need to pull them again

    Raises:
        ValueError: if the annotation format isn't what we expected

    Returns:
        tuple[list[str], list[SiltDetection]]: list of asset IDs that were pulled, and list of detections (in SILT-format) that were produced
    """
    # Instantiate the Kili client
    logger.info("Connecting to Kili...")
    kili = Kili(
        api_key=KILI_API_KEY,
        api_endpoint="https://cloud.eastus.kili-technology.com/api/label/v2/graphql",
    )
    logger.info("Connection successful")

    # Fetch assets with their latest labels
    # The 'latestLabel.jsonResponse' field retrieves the JSON content of the most recent label for each asset.
    detection_list = []
    new_asset_id_list = []
    logger.info("Downloading assets from Kili...")
    for asset in kili.assets(
        project_id=project_id,
        fields=[
            "id",
            "latestLabel.jsonResponse",
            "createdAt",
            "updatedAt",
            "metadata",
        ],
        asset_id_not_in=asset_id_list,
        status_in=["LABELED", "REVIEWED", "TO_REVIEW"],
        as_generator=True,
    ):
        latest_label_json = asset["latestLabel"]["jsonResponse"]
        logger.info(
            f"Processing {sum([len(frame['OBJECT_DETECTION_JOB']['annotations']) for frame in latest_label_json.values()])} annotations from asset {asset['id']}"
        )
        created_at = asset["createdAt"]
        updated_at = asset["updatedAt"]
        metadata = asset["metadata"]
        new_asset_id_list.append(asset["id"])

        # Now iterate through each frame
        for frame_idx in range(metadata["imageSetLength"]):
            kili_frame_annot_list = latest_label_json[str(frame_idx)][
                "OBJECT_DETECTION_JOB"
            ]["annotations"]

            # Need to convert to SILT annotations
            silt_frame_annot_list = []
            for kili_annot in kili_frame_annot_list:
                if len(kili_annot["boundingPoly"]) > 1:
                    raise ValueError("More than one bounding polygon")
                vertex_list = kili_annot["boundingPoly"][0]["normalizedVertices"]
                x_list = [v["x"] for v in vertex_list]
                y_list = [v["y"] for v in vertex_list]
                y_min = min(y_list)
                x_min = min(x_list)
                y_max = max(y_list)
                x_max = max(x_list)
                silt_frame_annot_list.append(
                    SiltObject(
                        type="box",
                        class_name="Satellite",
                        class_id=1,
                        y_min=y_min,
                        x_min=x_min,
                        y_max=y_max,
                        x_max=x_max,
                        y_center=(y_max + y_min) / 2.0,
                        x_center=(x_max + x_min) / 2.0,
                        bbox_height=y_max - y_min,
                        bbox_width=x_max - x_min,
                        source="turk_new",
                        correlation_id=kili_annot["mid"],
                        index=frame_idx,
                    )
                )

            # Convert to SILT-style detection
            detection_list.append(
                SiltDetection(
                    file=SiltFile(
                        filename=f"{metadata['imageSetId']}.{frame_idx}.fits",
                        id_sensor=metadata["idSensor"],
                    ),
                    sensor=SiltSensor(
                        width=metadata["frameWidthPixels"],
                        height=metadata["frameHeightPixels"],
                    ),
                    objects=silt_frame_annot_list,
                    index=frame_idx,
                    issues=[],  # TODO
                    approved=False,  # TODO
                    labeler_id="",  # TODO
                    request_size=metadata["imageSetLength"],
                    calibrations_used=False,
                    created=created_at,
                    updated=updated_at,
                    exp_start_time=metadata["expStartTime"],
                    image_set_id=metadata["imageSetId"],
                    sequence_id=frame_idx,
                    exposure=metadata["EXPTIME"],
                )
            )

    # Return what we pulled
    return new_asset_id_list, detection_list


def main(project_id: str, output_path: Path) -> None:
    # Read our log of already downloaded asset IDs
    path_to_asset_index = output_path / "downloaded_asset_ids.json"
    prev_asset_ids_list = []
    if os.path.exists(path_to_asset_index):
        with open(path_to_asset_index) as fp:
            prev_asset_ids_list = json.load(fp)

    # Request new annotations from Kili
    new_asset_id_list, detection_list = pull_new_annotations_for_project(
        project_id=project_id, asset_id_list=prev_asset_ids_list
    )

    # Write detections to disk
    os.makedirs(output_path, exist_ok=True)
    for detection in detection_list:
        filename = detection.file.filename.replace(".fits", ".json")
        with open(output_path / filename, "w") as fp:
            json.dump(detection.model_dump(mode="json"), fp=fp, indent=4)

    # Write the updated asset index to disk
    with open(path_to_asset_index, "w") as fp:
        json.dump(prev_asset_ids_list + new_asset_id_list, fp=fp, indent=4)


if __name__ == "__main__":
    logger.info("Starting script...")
    t0 = time.monotonic()

    # 1. Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Tool to download annotations for a project from Kili."
    )

    # 2. Add arguments
    # Positional argument (required)
    parser.add_argument(
        "project_id",
        type=str,
        help="Root of the dataset directory needing a catalog pulled",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Root of the dataset directory needing a catalog pulled",
    )

    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    main(project_id=args.project_id, output_path=Path(args.output_path))

    logger.info(f"Script complete in {time.monotonic() - t0:.2f} seconds")
