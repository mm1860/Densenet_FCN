#define COMPILER_MSVC
#define NOMINMAX

#include <string>
#include <vector>
#include <fstream>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/image_ops.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

void ReadImage(const vector<string> &file_names, const int height, const int width, vector<Tensor> &out_tensors)
{
	vector<Tensor> temp;
	for (auto &file_name : file_names)
	{
		ifstream file_in(file_name, ios::binary);
		Tensor image_tensor(DT_INT16, TensorShape({ 1, height, width, 1 }));
		short *image_data = image_tensor.flat<short>().data();
		file_in.read((char *)image_data, height * width * sizeof(short));
		file_in.close();

		temp.push_back(image_tensor);
	}

	// Construct graph to adjust window level and width
	Scope root = Scope::NewRootScope();
	auto input_ = Placeholder(root, DT_INT16, Placeholder::Shape({ 1, height, width, 1 }));
	auto cliped = ClipByValue(root, input_, (short)-70, (short)180);
	auto shifted = Sub(root, cliped, (short)-70);
	auto casted = Cast(root, shifted, DT_FLOAT);
	auto scaled = Div(root, casted, 250.0f);

	GraphDef graphdef;
	TF_CHECK_OK(root.ToGraphDef(&graphdef));
	ClientSession session(root);
	vector<Tensor> outputs;
	for (auto &img : temp)
	{
		TF_CHECK_OK(session.Run({ { input_, img} }, { scaled }, &outputs));
		out_tensors.push_back(outputs[0]);
		outputs.clear();
	}
}

void SaveImage(const vector<string> &names, const vector<Tensor> &masks)
{
	for (size_t i(0); i < names.size(); ++i)
	{
		const int *p = masks[i].flat<int>().data();
		auto shape = masks[i].shape().dim_sizes();
		int *arr = new int[shape[1] * shape[2]];
		memcpy(arr, p, shape[1] * shape[2] * sizeof(int));
		
		cv::Mat image_mat((int)shape[1], (int)shape[2], CV_32S, arr);
		image_mat.convertTo(image_mat, CV_8U, 255, 0);
		bool flag = cv::imwrite(names[i], image_mat);
		cout << names[i] << ": " << flag << endl;
		delete[] arr;
	}
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		cout << "ERROR: Need at least 3 parameters!" << endl;
		return -1;
	}

	// load model
	string model_path(argv[1]);
	GraphDef graphdef;
	Status load_status = ReadBinaryProto(Env::Default(), model_path, &graphdef);
	if (!load_status.ok())
	{
		cout << "ERROR: Loading model failed! " << model_path << endl;
		cout << load_status.ToString() << endl;
		return -1;
	}
	cout << "INFO: Model loaded." << endl;

	// import model to session
	SessionOptions options;
	unique_ptr<Session> session(NewSession(options));
	Status create_status = session->Create(graphdef);
	if (!create_status.ok())
	{
		cout << "ERROR: Creating graph in session failed! " << endl;
		cout << create_status.ToString() << endl;
		return -1;
	}
	cout << "INFO: Session successfully created." << endl;
	
	// read image
	vector<string> image_paths;
	for (int i(2); i < argc; ++i)
		image_paths.push_back(string(argv[i]));
	vector<Tensor> image_tensors;
	ReadImage(image_paths, 512, 512, image_tensors);

	vector<Tensor> outputs;
	vector<Tensor> masks;
	for (auto &image_tensor : image_tensors)
	{
		TF_CHECK_OK(session->Run({ {"Image", image_tensor} }, { "FCN-DenseNet/BinaryPred/Pred2Binary" }, {}, &outputs));
		masks.push_back(outputs[0]);
		outputs.clear();
	}
	cout << "INFO: Run inference finished." << endl;

	vector<string> out_paths;
	for (auto &path : image_paths)
		out_paths.push_back(string(path.begin(), path.end() - 3) + "jpg");
	
	SaveImage(out_paths, masks);
}