
#include "grid_subsampling.h"
// #include <iostream>


void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl)
{
	// int debug = 0

	// Initiate variables
	// ******************

	// Number of points in the cloud
	size_t N = original_points.size();

	// Dimension of the features
	size_t fdim = original_features.size() / N;

	// Limits of the cloud
	PointXYZ minCorner = min_point(original_points);
	PointXYZ maxCorner = max_point(original_points);
	PointXYZ originCorner = floor(minCorner * (1/sampleDl)) * sampleDl;  // align the origin with sampleDl

	// Dimensions of the grid
	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / sampleDl) + 1;
	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / sampleDl) + 1;
	//size_t sampleNZ = (size_t)floor((maxCorner.z - originCorner.z) / sampleDl) + 1;

	// if (debug) {
	// 	std::cout << "minCorner = " << minCorner.x << " " << minCorner.y << " " << minCorner.z << " " << std::endl;
	// 	std::cout << "maxCorner = " << maxCorner.x << " " << maxCorner.y << " " << maxCorner.z << " " << std::endl;
	// 	std::cout << "originCorner = " << originCorner.x << " " << originCorner.y << " " << originCorner.z << " " << std::endl;
	// 	std::cout << "sampleNX = " << sampleNX << " sampleNY = " << sampleNY << std::endl;
	// }

	// Check if features and classes need to be processed
	bool use_feature = original_features.size() > 0;
	bool use_classes = original_classes.size() > 0;


	// Create the sampled map
	// **********************

	// Verbose parameters
	int i = 0;
	int nDisp = N / 100;

	// Initiate variables
	size_t iX, iY, iZ, mapIdx;
	unordered_map<size_t, SampledData> data;

	for (auto& p : original_points)
	{
		// Position of point in sample map
		iX = (size_t)floor((p.x - originCorner.x) / sampleDl);
		iY = (size_t)floor((p.y - originCorner.y) / sampleDl);
		iZ = (size_t)floor((p.z - originCorner.z) / sampleDl);
		mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;

		// if (debug)
		// 	std::cout << "xyz = " << p.x << "\t" << p.y << "\t" << p.z << "\t" << "iX iY iZ = " << iX << " " << iY << " " << iZ << " ; mapIdx = "<< mapIdx << std::endl;

		// If not already created, create key
		if (data.count(mapIdx) < 1)
			data.emplace(mapIdx, SampledData(fdim));

		// Fill the sample map
		if (use_feature && use_classes)
			data[mapIdx].update_all(p, original_features.begin() + i * fdim, original_classes[i]);
		else if (use_feature)
			data[mapIdx].update_features(p, original_features.begin() + i * fdim);
		else if (use_classes)
			data[mapIdx].update_classes(p, original_classes[i]);
		else
			data[mapIdx].update_points(p);

		// Display
		i++;
	}

	// Divide for barycentre and transfer to a vector
	subsampled_points.reserve(data.size());
	if (use_feature)
		subsampled_features.reserve(data.size() * fdim);
	if (use_classes)
		subsampled_classes.reserve(data.size());
	for (auto& v : data)
	{
		subsampled_points.push_back(v.second.point * (1.0 / v.second.count));
		if (use_feature)
		{
		    float count = (float)v.second.count;
		    transform(v.second.features.begin(),
                      v.second.features.end(),
                      v.second.features.begin(),
                      [count](float f) { return f / count;});
            subsampled_features.insert(subsampled_features.end(),v.second.features.begin(),v.second.features.end());
		}
		if (use_classes)
			subsampled_classes.push_back(max_element(v.second.labels.begin(), v.second.labels.end(),
			[](const pair<int, int>&a, const pair<int, int>&b){return a.second < b.second;})->first);
	}
	return;
}



void batch_grid_subsampling(vector<PointXYZ>& original_points,
                              vector<PointXYZ>& subsampled_points,
                              vector<float>& original_features,
                              vector<float>& subsampled_features,
                              vector<int>& original_classes,
                              vector<int>& subsampled_classes,
                              vector<int>& original_batches,
                              vector<int>& subsampled_batches,
                              float sampleDl)
{
	// Initiate variables
	// ******************

	int b = 0;
	int sum_b = 0;

	// Loop over batches
	// *****************

	for (b = 0; b < original_batches.size(); b++)
	{
	    // Extract batch points
	    vector<PointXYZ> b_original_points = vector<PointXYZ>(original_points.begin () + sum_b,
	                                                          original_points.begin () + sum_b + original_batches[b]);

        // Create result containers
        vector<PointXYZ> b_subsampled_points;
        vector<float> b_subsampled_features;
        vector<int> b_subsampled_classes;

        // Compute subsampling on current batch
        grid_subsampling(b_original_points,
                         b_subsampled_points,
                         original_features,
                         b_subsampled_features,
                         original_classes,
                         b_subsampled_classes,
                         sampleDl);

        // Stack batches points
        subsampled_points.insert(subsampled_points.end(), b_subsampled_points.begin(), b_subsampled_points.end());

        // Stack new batch lengths
        subsampled_batches.push_back(b_subsampled_points.size());
        sum_b += original_batches[b];
	}

	return;
}