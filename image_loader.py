"""
Enhanced Image Loader Module
Handles loading DICOM and NIfTI medical image files with NIfTI to DICOM conversion
FIXED: DICOM series orientation issue
"""

import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import SimpleITK as sitk
from datetime import datetime
import tempfile
import shutil

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class MedicalImageLoader:
    """Class to load and process medical images (DICOM, NIfTI) with VTK support"""
    
    def __init__(self):
        self.image_data = None
        self.image_array = None
        self.metadata = {}
        self.file_type = None
        self.dimensions = None
        self.spacing = [1.0, 1.0, 1.0]
        self.origin = [0.0, 0.0, 0.0]
        self.temp_dicom_dir = None
        self.is_oriented_to_lps = False
        
    def load_file(self, file_path):
        """Load a medical image file (DICOM or NIfTI)"""
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_path.lower().endswith('.nii.gz'):
            file_extension = '.nii.gz'
        
        try:
            if file_extension == '.dcm':
                return self._load_dicom(file_path)
            elif file_extension in ['.nii', '.nii.gz']:
                return self._load_nifti_and_convert(file_path)
            else:
                return False, f"Unsupported file format: {file_extension}"
        except Exception as e:
            return False, f"Error loading file: {str(e)}"
    
    def normalize_orientation(self):
        if self.image_array is None:
            return False, "No image data to normalize"

        if hasattr(self, "sitk_image"):
            direction = np.array(self.sitk_image.GetDirection()).reshape(3, 3)
        else:
            direction = np.eye(3)

        flip_axes = [i for i in range(3) if direction[i, i] < 0]

        for axis in flip_axes:
            self.image_array = np.flip(self.image_array, axis=axis)

        return True, f"Orientation normalized (flipped axes: {flip_axes})"
    
    def _load_dicom(self, file_path):
        """Load a single DICOM file with adaptive normalization."""
        try:
            dicom_data = pydicom.dcmread(file_path, force=True)
            self.image_data = dicom_data
            self.file_type = 'DICOM'

            pixel_array = dicom_data.pixel_array.astype(np.float32)

            slope = getattr(dicom_data, 'RescaleSlope', 1)
            intercept = getattr(dicom_data, 'RescaleIntercept', 0)
            pixel_array = pixel_array * slope + intercept

            photometric = getattr(dicom_data, 'PhotometricInterpretation', '').upper()
            if photometric == 'MONOCHROME1':
                pixel_array = np.max(pixel_array) - pixel_array

            min_val, max_val = np.percentile(pixel_array, (0.5, 99.5))
            pixel_array = np.clip(pixel_array, min_val, max_val)
            pixel_array = (pixel_array - min_val) / (max_val - min_val + 1e-8)

            if pixel_array.ndim == 2:
                pixel_array = pixel_array[np.newaxis, :, :]

            self.image_array = pixel_array
            self.dimensions = self.image_array.shape
            self.is_oriented_to_lps = False

            self.metadata = {
                'PatientName': str(dicom_data.get('PatientName', 'Unknown')),
                'StudyDate': str(dicom_data.get('StudyDate', 'Unknown')),
                'Modality': str(dicom_data.get('Modality', 'Unknown')),
                'Rows': dicom_data.Rows,
                'Columns': dicom_data.Columns,
                'PhotometricInterpretation': photometric,
                'RescaleSlope': slope,
                'RescaleIntercept': intercept
            }

            return True, f"DICOM loaded successfully. Shape: {self.dimensions}"

        except Exception as e:
            return False, f"DICOM loading error: {str(e)}"

    
    def _load_nifti_and_convert(self, file_path):
        """Load NIfTI file using SimpleITK."""
        try:
            sitk_img = sitk.ReadImage(file_path)
            sitk_img = sitk.DICOMOrient(sitk_img, 'LPS')

            self.image_data = sitk_img
            self.file_type = 'NIfTI'

            self.image_array = sitk.GetArrayFromImage(sitk_img).astype(float)
            self.dimensions = self.image_array.shape

            self.spacing = list(map(float, sitk_img.GetSpacing()))
            self.origin = list(map(float, sitk_img.GetOrigin()))
            self.direction = sitk_img.GetDirection()
            self.is_oriented_to_lps = True

            self.metadata = {
                'Dimensions': str(self.dimensions),
                'VoxelSize': str(self.spacing),
                'Origin': str(self.origin),
                'Direction': str(self.direction),
                'DataType': sitk_img.GetPixelIDTypeAsString(),
            }

            return True, f"NIfTI loaded successfully with SimpleITK. Shape: {self.dimensions}"

        except Exception as e:
            return False, f"NIfTI loading error (SimpleITK): {str(e)}"

    def _normalize_orientation(self, image_array, direction_matrix):
        """Normalize image orientation to RAS+ convention."""
        try:
            dir = np.array(direction_matrix)
            flip_axes = []

            for i in range(3):
                if dir[i, i] < 0:
                    flip_axes.append(i)

            for axis in flip_axes:
                image_array = np.flip(image_array, axis=axis)

            if dir[2, 2] < 0:
                image_array = np.flip(image_array, axis=0)

            return image_array

        except Exception as e:
            print(f"[WARN] Orientation normalization failed: {e}")
            return image_array


    def get_dicom_directory(self):
        """Get the directory containing DICOM files"""
        return self.temp_dicom_dir
    
    def load_dicom_series(self, directory_path):
        """Load a DICOM series with consistent orientation."""
        try:
            # Collect DICOM files
            dicom_files = []
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith(('.dcm', '.ima', '.dicom')):
                        dicom_files.append(os.path.join(root, file))

            if len(dicom_files) == 0:
                return False, "No DICOM files found in directory"

            dicom_files.sort()

            # Load using SimpleITK
            try:
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(dicom_files)
                image = reader.Execute()

                # DON'T orient - keep original orientation
                # image = sitk.DICOMOrient(image, 'LPS')

                # Extract data
                self.image_array = sitk.GetArrayFromImage(image).astype(np.float32)
                self.spacing = list(image.GetSpacing())
                self.origin = list(image.GetOrigin())
                direction = np.array(image.GetDirection()).reshape(3, 3)

                self.sitk_image = image
                # Mark as NOT oriented since we kept original
                self.is_oriented_to_lps = False

            except Exception as e:
                print(f"[WARN] SimpleITK failed: {e}\nFalling back to manual loading.")
                # Fallback
                slices = []
                for file_path in dicom_files:
                    try:
                        ds = pydicom.dcmread(file_path, force=True)
                        if hasattr(ds, 'pixel_array'):
                            slice_data = ds.pixel_array.astype(np.float32)
                            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                                slice_data = slice_data * ds.RescaleSlope + ds.RescaleIntercept
                            slices.append(slice_data)

                            if len(slices) == 1 and hasattr(ds, 'PixelSpacing'):
                                self.spacing = [
                                    float(ds.PixelSpacing[0]),
                                    float(ds.PixelSpacing[1]),
                                    float(getattr(ds, 'SliceThickness', 1.0))
                                ]
                                orientation = getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
                                direction = np.array(orientation).reshape(2, 3)
                                normal = np.cross(direction[0], direction[1])
                                direction_matrix = np.vstack([direction[0], direction[1], normal])
                                direction_matrix = np.array(direction_matrix)
                    except Exception:
                        continue

                if len(slices) == 0:
                    return False, "No valid DICOM images found"

                self.image_array = np.stack(slices, axis=0)
                self.image_array = self._normalize_orientation(self.image_array, direction_matrix)
                self.is_oriented_to_lps = False

            self.file_type = 'DICOM_SERIES'
            self.dimensions = self.image_array.shape
            self.temp_dicom_dir = directory_path

            self.metadata = {
                'NumSlices': len(dicom_files),
                'Dimensions': str(self.dimensions),
                'Spacing': str(self.spacing),
                'Origin': str(self.origin),
                'Directory': directory_path
            }

            return True, f"DICOM series loaded: {len(dicom_files)} files. Shape: {self.dimensions}"

        except Exception as e:
            return False, f"DICOM series loading error: {str(e)}"

    
    def get_slice(self, orientation, slice_number):
        """
        Return RAW slice without ANY flipping - let GUI handle it
        """
        if self.image_array is None:
            return None

        try:
            orientation = orientation.lower()
            
            if orientation == 'axial':
                slice_number = min(slice_number, self.dimensions[0] - 1)
                slice_2d = self.image_array[slice_number, :, :]

            elif orientation == 'coronal':
                slice_number = min(slice_number, self.dimensions[1] - 1)
                slice_2d = self.image_array[:, slice_number, :]

            elif orientation == 'sagittal':
                slice_number = min(slice_number, self.dimensions[2] - 1)
                slice_2d = self.image_array[:, :, slice_number]

            else:
                return None

            return slice_2d

        except Exception as e:
            print(f"Error getting slice: {e}")
            return None

    def get_num_slices(self, orientation):
        """Get number of slices for given orientation"""
        if self.image_array is None:
            return 0
        
        if orientation.lower() == 'axial':
            return self.dimensions[0]
        elif orientation.lower() == 'sagittal':
            return self.dimensions[2]
        elif orientation.lower() == 'coronal':
            return self.dimensions[1]
        else:
            return 0
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dicom_dir and os.path.exists(self.temp_dicom_dir):
            try:
                shutil.rmtree(self.temp_dicom_dir)
                self.temp_dicom_dir = None
            except:
                pass
    
    def __del__(self):
        """Destructor to clean up temp files"""
        self.cleanup()