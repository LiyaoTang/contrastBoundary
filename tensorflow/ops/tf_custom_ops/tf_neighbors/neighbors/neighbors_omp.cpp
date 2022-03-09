
#include <numeric>
#include "neighbors.h"
#include "omp.h"

void batch_nanoflann_neighbors_omp(vector<PointXYZ>& queries,  // M
                                   vector<PointXYZ>& supports,  // N
                                   vector<int>& q_batches,  // batches len - queries
                                   vector<int>& s_batches,  // batches len - supports
                                   vector<int>& neighbors_indices,
                                   float radius,
								   bool sorted)
{

	// Initiate variables
	// ******************

	// indices
	int i0 = 0;

	// Square radius
	float r2 = radius * radius;

	// Counting vector
	float d2;
	vector<vector<pair<size_t, float>>> all_inds_dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;

	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud current_cloud;

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
                                                        PointCloud,
                                                        3 > my_kd_tree_t;

	// Search neigbors indices
	// ***********************

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = sorted;

	// Batches
	vector<int> q_starts(q_batches.size());
	vector<int> s_starts(s_batches.size());
	vector<int> max_counts(q_batches.size());
	std::partial_sum(q_batches.begin(), q_batches.end(), q_starts.begin());
	std::partial_sum(s_batches.begin(), s_batches.end(), s_starts.begin());

	// parallel across batches
# pragma omp parallel for
	for (int b = 0; b < q_batches.size(); b++) {

		// current support cloud
		PointCloud support_cloud;
		support_cloud.pts = vector<PointXYZ>(supports.begin() + s_starts[b], supports.begin() + s_starts[b] + s_batches[b]);

		// build tree
		my_kd_tree_t* index = new my_kd_tree_t(3, support_cloud, tree_params);
	    index->buildIndex();

	    // Initial guess of neighbors size
		int max_count = 32;

		// current query cloud
		int i0 = queries.begin() + q_starts[b];
		const &vector<PointXYZ> query_cloud(queries.begin() + q_starts[b], queries.begin() + q_starts[b] + q_batches[b]);

		// iterate over current queries
		for (auto& p0 : query_cloud) {
			all_inds_dists[i0].reserve(max_count);

			// Find neighbors
			float query_pt[3] = { p0.x, p0.y, p0.z};
			size_t nMatches = index->radiusSearch(query_pt, r2, all_inds_dists[i0], search_params);

			// Update max count
			if (nMatches > max_count)
				max_count = nMatches;

			// Increment query idx
			i0++;
		}

		// store the max counts
		max_counts[b] = max_count
		delete index;

	}

	// Reserve the memory
	int max_count = max_element(max_counts.begin(), max_counts.end());
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	for (auto& inds_dists : all_inds_dists)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

		for (int j = 0; j < max_count; j++)
		{
			if (j < inds_dists.size())
				neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	return;
}



void cpp_knn_batch_omp(const float* batch_data, const size_t batch_size, const size_t npts, const size_t dim, 
				const float* queries, const size_t nqueries,
				const size_t K, long* batch_indices){

# pragma omp parallel for
	for(size_t bid=0; bid < batch_size; bid++){

		const float* points = &batch_data[bid*npts*dim];
		long* indices = &batch_indices[bid*nqueries*K];

		// create the kdtree
		typedef KDTreeTableAdaptor< float, float> KDTree;
		KDTree mat_index(npts, dim, points, 10);
		
		mat_index.index->buildIndex();

		std::vector<float> out_dists_sqr(K);
		std::vector<size_t> out_ids(K);

		// iterate over the points
		for(size_t i=0; i<nqueries; i++){
			nanoflann::KNNResultSet<float> resultSet(K);
			resultSet.init(&out_ids[0], &out_dists_sqr[0] );
			mat_index.index->findNeighbors(resultSet, &queries[bid*nqueries*dim + i*dim], nanoflann::SearchParams(10));
			for(size_t j=0; j<K; j++){
				indices[i*K+j] = long(out_ids[j]);
			}
		}

	}

}