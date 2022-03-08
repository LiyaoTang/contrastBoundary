#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "neighbors/neighbors.h"

using namespace tensorflow;

REGISTER_OP("BatchOrderedNeighbors")
    .Input("queries: float")
    .Input("supports: float")
    .Input("q_batches: int32")
    .Input("s_batches: int32")
    .Input("radius: float")
    .Output("neighbors: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

        // Create input shape container
        ::tensorflow::shape_inference::ShapeHandle input;

        // Check inputs rank
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input));

        // Create the output shape
        c->set_output(0, c->UnknownShapeOfRank(2));

        return Status::OK();
    });





class BatchOrderedNeighborsOp : public OpKernel {
    public:
    explicit BatchOrderedNeighborsOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

        // Grab the input tensors
        const Tensor& queries_tensor = context->input(0);
        const Tensor& supports_tensor = context->input(1);
        const Tensor& q_batches_tensor = context->input(2);
        const Tensor& s_batches_tensor = context->input(3);
        const Tensor& radius_tensor = context->input(4);

        // check shapes of input and weights
        const TensorShape& queries_shape = queries_tensor.shape();
        const TensorShape& supports_shape = supports_tensor.shape();
        const TensorShape& q_batches_shape = q_batches_tensor.shape();
        const TensorShape& s_batches_shape = s_batches_tensor.shape();

        // check input are [N x 3] matrices
        DCHECK_EQ(queries_shape.dims(), 2);
        DCHECK_EQ(queries_shape.dim_size(1), 3);
        DCHECK_EQ(supports_shape.dims(), 2);
        DCHECK_EQ(supports_shape.dim_size(1), 3);

        // Check that Batch lengths are vectors and same number of batch for both query and support
        DCHECK_EQ(q_batches_shape.dims(), 1);
        DCHECK_EQ(s_batches_shape.dims(), 1);
        DCHECK_EQ(q_batches_shape.dim_size(0), s_batches_shape.dim_size(0));

        // Points Dimensions
        int Nq = (int)queries_shape.dim_size(0);
        int Ns = (int)supports_shape.dim_size(0);

        // Number of batches
        int Nb = (int)q_batches_shape.dim_size(0);

        // get the data as std vector of points
        float radius = radius_tensor.flat<float>().data()[0];
        vector<PointXYZ> queries = vector<PointXYZ>((PointXYZ*)queries_tensor.flat<float>().data(),
                                                    (PointXYZ*)queries_tensor.flat<float>().data() + Nq);
        vector<PointXYZ> supports = vector<PointXYZ>((PointXYZ*)supports_tensor.flat<float>().data(),
                                                     (PointXYZ*)supports_tensor.flat<float>().data() + Ns);

        // Batches lengths
        vector<int> q_batches = vector<int>((int*)q_batches_tensor.flat<int>().data(),
                                            (int*)q_batches_tensor.flat<int>().data() + Nb);
        vector<int> s_batches = vector<int>((int*)s_batches_tensor.flat<int>().data(),
                                            (int*)s_batches_tensor.flat<int>().data() + Nb);


        // Create result containers
        vector<int> neighbors_indices;

        // Compute results
        //batch_ordered_neighbors(queries, supports, q_batches, s_batches, neighbors_indices, radius);
        batch_nanoflann_neighbors(queries, supports, q_batches, s_batches, neighbors_indices, radius);

        // Maximal number of neighbors
        int max_neighbors = neighbors_indices.size() / Nq;

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(Nq);
        output_shape.AddDim(max_neighbors);

        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_tensor = output->matrix<int>();

        // Fill output tensor
        for (int i = 0; i < output->shape().dim_size(0); i++)
        {
            for (int j = 0; j < output->shape().dim_size(1); j++)
            {
                output_tensor(i, j) = neighbors_indices[max_neighbors * i + j];
            }
        }
    }
};


REGISTER_KERNEL_BUILDER(Name("BatchOrderedNeighbors").Device(DEVICE_CPU), BatchOrderedNeighborsOp);