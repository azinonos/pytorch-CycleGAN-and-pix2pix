import SimpleITK as sitk
import os, shutil

# Change root path to the directory where your data is stored
root_path = '/Users/azinonos/Desktop/BRATS_Sample'

def convertMhdToNii():
	count = 0

	for root, dirs, files in os.walk(root_path):
		for file in files:
			if file.endswith(".mha"):
				filename = os.path.splitext(file)[0]
				file_dir = os.path.join(root, file)
				nii_path = root + '/' + filename + '.nii'

				img = sitk.ReadImage(file_dir)
				sitk.WriteImage(img, nii_path)
				count += 1
				print("Converted:", file)

	print("Converted {} files".format(count))

def createABDomains(MRI_type, delete=False):
	trainA_dir = root_path + '/trainA/'
	trainB_dir = root_path + '/trainB/'

	# Create directories if they don't exist
	if not os.path.exists(trainA_dir):
		os.makedirs(trainA_dir)
	if not os.path.exists(trainB_dir):
		os.makedirs(trainB_dir)

	# Go through data and split them into trainA and trainB domains
	for root, dirs, files in os.walk(root_path):
		for file in files:
			# Move MRI image - Domain A
			if MRI_type in file and file.endswith(".mha"):
				print("Moving:", file)
				file_dir = os.path.join(root, file)
				os.rename(file_dir, os.path.join(trainA_dir, file))

			# Move MRI label - Domain B
			if '3more' in file and file.endswith(".mha"):
				print("Moving:", file)
				file_dir = os.path.join(root, file)
				os.rename(file_dir, os.path.join(trainB_dir, file))

	# Delete old folders
	print("-"*10)
	if delete:
		for root, dirs, files in os.walk(root_path):
			for d in dirs:
				full_dir = os.path.join(root,d)
				if 'trainA' not in full_dir and 'trainB' not in full_dir:
					print("Deleted:", full_dir)
					shutil.rmtree(full_dir)

if __name__ == "__main__":
	createABDomains('T1c', delete=True)