#include "host.h"

void datasetBinaryRead(vector<int> &data, string path){
	ifstream file_(path, ios::in | ios::binary);
	int32_t size;
	file_.read((char*)&size, sizeof(size));
	int32_t temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}
    string binaryFile = argv[1];
    cl_int err;
    unsigned fileBufSize;

	auto t_start = chrono::high_resolution_clock::now();
   
	vector<int> X_train;
	vector<int> y_train;
	
	datasetBinaryRead(X_train, X_train_path);
	datasetBinaryRead(y_train, y_train_path);

	int N_SAMPLE = y_train.size();
	int input_int = X_train.size();
	 
	vector<int, aligned_allocator<int>> input_gmem(input_int);
	for(int i = 0; i < input_int; i++){
		input_gmem[i] = X_train[i];
	}
	
	vector<int, aligned_allocator<int>> labels_gmem(N_SAMPLE);
	for(int i = 0; i < N_SAMPLE; i++){
		labels_gmem[i] = y_train[i];
	}

	//We need a seed ID. To generate in a random yet determenistic (for later debug purposes) fashion, we use bits of log2 as some random stuff.
	vector<int, aligned_allocator<int>> ID_gmem(Dhv/32);
	srand (time(NULL));
	for(int i = 0; i < Dhv/32; i++){
		long double temp = log2(i+2.5) * pow(2, 31);
		long long int temp2 = (long long int)(temp);
		temp2 = temp2 % int(pow(2, 31));
		ID_gmem[i] = int(temp2);
		//ID_gmem[i] = int(rand());
	}
	vector<int, aligned_allocator<int>> classHV_gmem(N_CLASS*Dhv);	
	vector<int, aligned_allocator<int>> trainScore(1);
	
	vector<int, aligned_allocator<int>> encHV_gmem((Dhv/32)*N_SAMPLE);

	auto t_elapsed = chrono::high_resolution_clock::now() - t_start;
	long mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
	long mSec_train = mSec;

// OPENCL HOST CODE AREA START
	
// ------------------------------------------------------------------------------------
// Step 1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
//	   Search for Platform: Xilinx 
// Check if the current platform matches Target_Platform_Vendor
// ------------------------------------------------------------------------------------	
    std::vector<cl::Device> devices = get_devices("Xilinx");
    devices.resize(1);
    cl::Device device = devices[0];

// ------------------------------------------------------------------------------------
// Step 1: Create Context
// ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	
// ------------------------------------------------------------------------------------
// Step 1: Create Command Queue
// ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

// ------------------------------------------------------------------
// Step 1: Load Binary File from disk
// ------------------------------------------------------------------		
    char* fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
	
// -------------------------------------------------------------
// Step 1: Create the program object from the binary and program the FPGA device with it
// -------------------------------------------------------------	
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

// -------------------------------------------------------------
// Step 1: Create Kernels
// -------------------------------------------------------------
    OCL_CHECK(err, cl::Kernel krnl_hd(program,"hd", &err));

// ================================================================
// Step 2: Setup Buffers and run Kernels
// ================================================================
//   o) Allocate Memory to store the results 
//   o) Create Buffers in Global Memory to store data
// ================================================================

// .......................................................
// Allocate Global Memory for sources
// .......................................................	
    OCL_CHECK(err, cl::Buffer buf_input      (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*input_gmem.size(),  input_gmem.data(), &err));
	OCL_CHECK(err, cl::Buffer buf_ID         (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*ID_gmem.size(), ID_gmem.data(), &err));
	OCL_CHECK(err, cl::Buffer buf_classHV    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*classHV_gmem.size(), classHV_gmem.data(), &err));
	OCL_CHECK(err, cl::Buffer buf_encHV    (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*encHV_gmem.size(), encHV_gmem.data(), &err));
	OCL_CHECK(err, cl::Buffer buf_labels     (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*labels_gmem.size(), labels_gmem.data(), &err));
	OCL_CHECK(err, cl::Buffer buf_trainScore (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*trainScore.size(), trainScore.data(), &err));

// ============================================================================
// Step 2: Set Kernel Arguments and Run the Application
//         o) Set Kernel Arguments
//         o) Copy Input Data from Host to Global Memory on the device
//         o) Submit Kernels for Execution
//         o) Copy Results from Global Memory, device to Host
// ============================================================================	

	OCL_CHECK(err, err = krnl_hd.setArg(0, buf_input));
    OCL_CHECK(err, err = krnl_hd.setArg(1, buf_ID));
    OCL_CHECK(err, err = krnl_hd.setArg(2, buf_classHV));
    OCL_CHECK(err, err = krnl_hd.setArg(3, buf_labels));
    OCL_CHECK(err, err = krnl_hd.setArg(4, buf_encHV));
    OCL_CHECK(err, err = krnl_hd.setArg(5, buf_trainScore));
    OCL_CHECK(err, err = krnl_hd.setArg(6, train));
    OCL_CHECK(err, err = krnl_hd.setArg(7, N_SAMPLE));

// ------------------------------------------------------
// Step 2: Copy Input data from Host to Global Memory on the device
// ------------------------------------------------------
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buf_input, buf_ID, buf_labels}, 0 ));
	cout << "\nTrain data copied to the device global memory" << endl;	
// ----------------------------------------
// Step 2: Submit Kernels for Execution
// ----------------------------------------
	t_start = chrono::high_resolution_clock::now();
	
    OCL_CHECK(err, err = q.enqueueTask(krnl_hd));
	
// --------------------------------------------------
// Step 2: Copy Results from Device Global Memory to Host
// --------------------------------------------------
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buf_classHV, buf_trainScore}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
	t_elapsed = chrono::high_resolution_clock::now() - t_start;
	
	mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
	cout << "Reading train data took " << mSec_train << " mSec" << endl;
	cout << "Train execution took " << mSec << " mSec" << endl;
	
// OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
	
	for(int i = 0; i < N_CLASS; i++){
		//cout << classHV_gmem[i*Dhv] << "\t" << classHV_gmem[i*Dhv + Dhv - 1] << endl;
	}
	cout << "Train accuracy = " << float(trainScore[0])/N_SAMPLE << endl << endl;
	
	t_start = chrono::high_resolution_clock::now();
	vector<int> X_test;
	vector<int> y_test;
	
	datasetBinaryRead(X_test, X_test_path);
	datasetBinaryRead(y_test, y_test_path);

	input_int = X_test.size();	 
	input_gmem.resize(input_int);
	for(int i = 0; i < input_int; i++){
		input_gmem[i] = X_test[i];
	}
	
	 t_elapsed = chrono::high_resolution_clock::now() - t_start;
	mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
	long mSec_test = mSec;
	
	int N_TEST = y_test.size();
	labels_gmem.resize(N_TEST);
	
	OCL_CHECK(err, cl::Buffer buf_input2      (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*input_gmem.size(),  input_gmem.data(), &err));
	OCL_CHECK(err, cl::Buffer buf_labels2     (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*labels_gmem.size(), labels_gmem.data(), &err));

	OCL_CHECK(err, err = krnl_hd.setArg(0, buf_input2));
    OCL_CHECK(err, err = krnl_hd.setArg(3, buf_labels2));
    train = 0; //i.e., inference
    OCL_CHECK(err, err = krnl_hd.setArg(6, train));
    OCL_CHECK(err, err = krnl_hd.setArg(7, N_TEST));

	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buf_input2}, 0 ));	
	cout << "Test data copied to the device global memory" << endl;	
	t_start = chrono::high_resolution_clock::now();
	OCL_CHECK(err, err = q.enqueueTask(krnl_hd));
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buf_labels2}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    t_elapsed = chrono::high_resolution_clock::now() - t_start;
    mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
    cout << "Reading test data took " << mSec_test << " mSec" << endl;
	cout << "Test execution took " << mSec << " mSec" << endl;
    
    int correct = 0;
    for(int i = 0; i < N_TEST; i++)
    	if(labels_gmem[i] == y_test[i])
    		correct += 1;
    cout << "Test accuracy = " << float(correct)/N_TEST << endl;
	
// ============================================================================
// Step 3: Release Allocated Resources
// ============================================================================
    delete[] fileBuf;

}

