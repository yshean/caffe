// Copyright 2014 BVLC and contributors.
//
// This program takes in a trained network and an input blob, and then dumps
// all the intermediate blobs produced by the net to individual binary
// files stored in protobuffer binary formats.
// Usage:
//    dump_network input_net_param trained_net_param \
//        input_blob output_prefix 0/1
// if input_net_param is 'none', we will directly load the network from
// trained_net_param.

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "fcntl.h"
#include "google/protobuf/text_format.h"

//
#include <fstream>
#include <iostream>
#include <sstream>
//

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
const int ARG_MIN = 5;
int main(int argc, char** argv) {

    if (argc < ARG_MIN)
    {
        LOG(ERROR) << "Usage: " << argv[0] << " input_net_param trained_net_param input_blob output_prefix 0/1 [0 = matlab, 1 = cpp]";
        return 1;
    }
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_phase(Caffe::TEST);

    shared_ptr<Net<float> > caffe_net;
    if (strcmp(argv[1], "none") == 0)
    {
        // We directly load the net param from trained file
        caffe_net.reset(new Net<float>(argv[2]));
    } else {
        caffe_net.reset(new Net<float>(argv[1]));
    }
    caffe_net->CopyTrainedLayersFrom(argv[2]);

    vector<Blob<float>* > input_vec;
    shared_ptr<Blob<float> > input_blob(new Blob<float>());
    if (strcmp(argv[3], "none") != 0)
    {
        BlobProto input_blob_proto;
        ReadProtoFromBinaryFile(argv[3], &input_blob_proto);
        input_blob->FromProto(input_blob_proto);
        input_vec.push_back(input_blob.get());
        LOG(ERROR) << "Read ok ";
    }

    string output_prefix(argv[4]);
    string output_prefix_txt = output_prefix;// + "text";
    // Run the network without training.
    LOG(ERROR) << "Performing Forward";
    caffe_net->Forward(input_vec);
    LOG(ERROR) << "OK! ";
    // Now, let's dump all the layers

    const vector<string>& blob_names = caffe_net->blob_names();
    const vector<shared_ptr<Blob<float> > >& blobs = caffe_net->blobs();
    for (int blobid = 0; blobid < caffe_net->blobs().size(); ++blobid) {
        // Serialize blob
        if ((blobid == 10) || (blobid ==13) || (blobid ==15))
        {
            LOG(ERROR) << "Dumping " << blob_names[blobid];
            BlobProto output_blob_proto;
            blobs[blobid]->ToProto(&output_blob_proto);
            //    WriteProtoToBinaryFile(output_blob_proto,
            //        output_prefix + blob_names[blobid]);
            //    WriteProtoToTextFile(output_blob_proto,
            //        output_prefix_txt + blob_names[blobid]);

            std::string fname = output_prefix_txt + blob_names[blobid];
            std::string fname_format = fname + "form";

            std::cout << fname << std::endl;

            std::string out_str;
            WriteProtoToString(output_blob_proto, out_str);
            std::stringstream out_stream(out_str);
            std::ofstream out_file_str(fname.c_str());


            if (out_file_str.is_open())
            {
                int n_img, width,height, channels;
                std::string temp_str;
                out_stream >> temp_str;
                out_stream >> n_img;

                out_stream >> temp_str;
                out_stream >> channels;

                out_stream >> temp_str;
                out_stream >> height;

                out_stream >> temp_str;
                out_stream >> width;

                std::ofstream out_file_format_str(fname_format.c_str());
                if (out_file_format_str.is_open())
                {
                    out_file_format_str << n_img << " " << channels << " " << height << " " << width << std::endl;
                }
                out_file_format_str.close();
                int one_img_length = channels * width * height;
                double data;

                if (argc > 5 && strcmp(argv[5], "1") == 0)
                {
                    out_file_str << n_img << " " << channels << " " << height << " " << width << std::endl;
                }
                for (int i=0; i < n_img ;++i)
                {
                    for (int j=0; j < one_img_length; ++j)
                    {
                        out_stream >> temp_str;
                        out_stream >> data;

                        out_file_str << data << " ";
                    }
                    out_file_str << std::endl;
                }
            }
            out_file_str.close();
        }

    }

    return 0;
}
