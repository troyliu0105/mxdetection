from gluoncv.data import VOCDetection, RecordFileDetection

from mxdetection.datasets.builder import DATASETS

DATASETS.register_module("VOCDetection", module=VOCDetection)
DATASETS.register_module("VOCDetectionRecordFile", module=RecordFileDetection)
