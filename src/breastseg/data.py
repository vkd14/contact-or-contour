import zipfile
from pathlib import Path
import hashlib
import cv2
import numpy as np
from .core import binary_mask, validate_split


def load_bbbc039(root, splits=("train","val","test")):
    root=Path(root)
    iz,mz,metadata=(zipfile.ZipFile(root/name) for name in ("images.zip","masks.zip","metadata.zip"))
    cases=[]
    filenames={"train":"training.txt","val":"validation.txt","test":"test.txt"}
    splits=tuple(splits)
    if not splits or not set(splits).issubset(filenames) or len(set(splits))!=len(splits):
        raise ValueError("splits must be a nonempty unique subset of train/val/test")
    for split in splits:
        filename=filenames[split]
        for name in sorted(metadata.read("metadata/"+filename).decode().splitlines()):
            sid=Path(name.strip()).stem
            if not sid:
                continue
            raw=cv2.imdecode(np.frombuffer(iz.read("images/"+sid+".tif"),np.uint8),-1)
            mask=cv2.imdecode(np.frombuffer(mz.read("masks/"+sid+".png"),np.uint8),1)
            if raw is None or mask is None:
                raise ValueError("Unreadable image/mask "+sid)
            gt=binary_mask(mask[...,2])
            if raw.shape!=gt.shape:
                raise ValueError("Mismatched geometry "+sid)
            digest=hashlib.sha256(raw.tobytes()).hexdigest()
            cases.append(dict(id=sid,group=sid,split=split,image_sha256=digest,image=raw,mask=gt))
    validate_split(cases)
    expected={"train":100,"val":50,"test":50}
    counts={s:sum(c["split"]==s for c in cases) for s in splits}
    if counts!={s:expected[s] for s in splits}:
        raise ValueError(f"Unexpected BBBC039 split counts: {counts}")
    return cases


def _load_nucleusseg(root, extended=False):
    """Load an official Bilkent NucleusSegData release and fixed split."""
    root = Path(root)
    release_directory = "NucleusSegDataExtended" if extended else "NucleusSegData"
    if (root / release_directory).is_dir():
        root = root / release_directory
    specifications = (
        ("TrainingSet", "tr", "tr_ann", "train", "training"),
        ("Huh7TestSet", "huh7_ts", "huh7_ts_ann", "test", "Huh7"),
        ("HepG2TestSet", "hepg2_ts", "hepg2_ts_ann", "test", "HepG2"),
    )
    cases = []
    for directory, image_prefix, annotation_prefix, split, subset in specifications:
        folder = root / directory
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing NucleusSegData directory: {folder}")
        images = sorted(folder.glob(f"{image_prefix}*.jpg"),
                        key=lambda path: int(path.stem.removeprefix(image_prefix)))
        for image_path in images:
            index = int(image_path.stem.removeprefix(image_prefix))
            annotation_path = (folder / f"{image_path.stem}.gold" if extended
                               else folder / f"{annotation_prefix}{index}")
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None or not annotation_path.is_file():
                raise ValueError(f"Unreadable image/annotation pair: {image_path}")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if extended:
                with annotation_path.open() as handle:
                    declared_shape = tuple(int(value) for value in handle.readline().split())
                if len(declared_shape) != 2:
                    raise ValueError(f"Invalid annotation header: {annotation_path}")
                instances = np.loadtxt(annotation_path, dtype=np.int32, skiprows=1)
                if instances.shape != declared_shape:
                    raise ValueError(f"Annotation header/contents mismatch: {annotation_path}")
            else:
                instances = np.loadtxt(annotation_path, dtype=np.int32)
            if instances.shape != image.shape[:2] or (instances < 0).any():
                raise ValueError(f"Invalid annotation geometry: {annotation_path}")
            expected = np.arange(int(instances.max()) + 1)
            if not np.array_equal(np.unique(instances), expected):
                raise ValueError(f"Non-contiguous instance identifiers: {annotation_path}")
            identifier = f"{subset.lower()}_{index:02d}"
            cases.append({
                "id": identifier,
                "group": identifier,
                "split": split,
                "subset": subset,
                "dataset_version": "extended" if extended else "original",
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "image": image,
                "instances": instances,
                "mask": (instances > 0).astype(np.uint8),
            })
    validate_split(cases)
    counts = {
        "train_images": sum(case["split"] == "train" for case in cases),
        "Huh7_images": sum(case["subset"] == "Huh7" for case in cases),
        "HepG2_images": sum(case["subset"] == "HepG2" for case in cases),
        "train_nuclei": sum(int(case["instances"].max()) for case in cases if case["split"] == "train"),
        "Huh7_nuclei": sum(int(case["instances"].max()) for case in cases if case["subset"] == "Huh7"),
        "HepG2_nuclei": sum(int(case["instances"].max()) for case in cases if case["subset"] == "HepG2"),
    }
    expected = (
        {"train_images": 25, "Huh7_images": 11, "HepG2_images": 25,
         "train_nuclei": 1126, "Huh7_nuclei": 891, "HepG2_nuclei": 1312}
        if extended else
        {"train_images": 10, "Huh7_images": 11, "HepG2_images": 16,
         "train_nuclei": 785, "Huh7_nuclei": 891, "HepG2_nuclei": 985}
    )
    if counts != expected:
        raise ValueError(f"Unexpected NucleusSegData composition: {counts}")
    return cases


def load_nucleusseg(root):
    """Load the original 37-image NucleusSegData benchmark.

    This release has 10 training, 11 Huh7 test, and 16 HepG2 test images.
    """
    return _load_nucleusseg(root, extended=False)


def load_nucleusseg_extended(root):
    """Load the extended 61-image NucleusSegData/AttentionBoost benchmark.

    This release has 25 training, 11 Huh7 test, and 25 HepG2 test images.
    Its ``.gold`` annotation files begin with a declared matrix shape.
    """
    return _load_nucleusseg(root, extended=True)
