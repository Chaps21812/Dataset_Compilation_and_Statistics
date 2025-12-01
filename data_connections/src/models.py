from datetime import datetime

from pydantic import BaseModel


class KiliDetection(BaseModel):
    OBJECT_DETECTION_JOB: KiliObjectDetectionJob
    ANNOTATION_JOB_COUNTER: dict[str, str]
    ANNOTATION_NAMES_JOB: dict[str, str]


class KiliObjectDetectionJob(BaseModel):
    annotations: list[KiliAnnotation]


class KiliAnnotation(BaseModel):
    children: dict
    isKeyFrame: bool
    categories: list
    mid: str
    type: str
    boundingPoly: list[KiliPoly]


class KiliPoly(BaseModel):
    normalizedVertices: list[KiliVertex]


class KiliVertex(BaseModel):
    x: float
    y: float


class SiltDetection(BaseModel):
    file: SiltFile
    sensor: SiltSensor
    objects: list[SiltObject]
    issues: list[str]
    index: int
    approved: bool
    labeler_id: str
    request_id: int = None
    image_id: str = None
    request_size: int
    calibrations_used: bool
    created: datetime
    updated: datetime
    exp_start_time: datetime
    image_set_id: str
    sequence_id: int
    exposure: float
    astro_fit_validated: bool = None
    empty_image: bool = None
    too_few_stars: bool = None


class SiltFile(BaseModel):
    filename: str
    id_sensor: str
    empty_image: bool = None
    too_few_stars: bool = None


class SiltSensor(BaseModel):
    width: int
    height: int


class SiltObject(BaseModel):
    type: str
    class_name: str
    class_id: int
    y_min: float
    x_min: float
    y_max: float
    x_max: float
    y_center: float
    x_center: float
    bbox_height: float
    bbox_width: float
    source: str
    magnitude: float = None
    correlation_id: str
    index: int
    iso_flux: int = None
