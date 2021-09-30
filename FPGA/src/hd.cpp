
#include "hd.h"

/*
 * inputStream fetches input features as ints, and streames to the next functions.
 *
 * input_gmem (input): input data port; each feature is quantized to an integer.
 * feature_stream (output): N_FEAT_PAD parallel streams to stream the data to the next module.
 * size (input): number of data sampels.
 */



void inputStream(int *input_gmem, hls::stream<int> feature_stream[N_FEAT_PAD], int size){

	loop_inputs:
	for(int iter_read = 0; iter_read < size; iter_read++){
		#pragma HLS LOOP_TRIPCOUNT MIN=1000 MAX=1000
		 //Need to move the pointer by intPerInput after each input
		int offset = iter_read * N_FEAT;
		loop_features:
		for(int i = 0; i < N_FEAT; i++){
			feature_stream[i] << input_gmem[offset + i];
		}
		for(int i = 0; i < PAD; i++){
			feature_stream[N_FEAT + i] << 0;
		}
	}

}

/*
 * encodeUnit reads input features from the stream and obtains encoding hypervector using random projection (RP) algorithm.
 * RP is all about a matrix-vector multplicatoin. The matrix is ID hypervectors arrangaed as Dhv*Div (Dhv = HV dimensions, Div = #input vector elements).
 * We break this A=(Dhv*Div) * B=(Div*1) multiplicaton into slinding windows of ROW*COL on the matrix A.
 * It takes Div/COL cycles for the sliding window to reach the right-most column, when we accomplish ROW dimensions. Total cycles = (Dhv*Div)/(ROW*COL)*(latency of accumulating COL partials).
 * Note that we only have a single seed ID (Dhv bits) in a (Dhv/ROW) x ROW array for column 0, and generate the ID of column k by k-bit rotating (circular shift).
 *
 * feature_stream (input): N_FEAT_PAD parallel streams from the previous module to read input features of a data sample.
 * ID (input): seed ID hypervector, packed as Dhv/ROW (rows) of ROW bit (total Dhv bits).
 * enc_stream (output): streams ROW encoded dimensions per (Div/COL) cycles to the next module.
 * size (input): number of data samples.
 *
 */
template<int ROW_, int COL_>
void encodeUnit(hls::stream<int> feature_stream[N_FEAT_PAD], ap_int<ROW> ID[Dhv/ROW], hls::stream<dt_int2> enc_stream[ROW], int size){

	//Operate on ROW encoding dimension per cycle
	int encHV_partial[ROW];
	#pragma HLS array_partition variable=encHV_partial

	int feature_array[N_FEAT_PAD];
	//Factor the feature memory into COL, as we read COL elements of it in parallel.
	#pragma HLS array_partition variable=feature_array cyclic factor=COL_

	//ID register to keep ROW+COL bits for a ROW*COL window.
	//ID memory has ROW bits per cell, so we use 2*ROW bit register (extra bits will be used in the next window).
	//It might look a little tricky. See the report for visualization.
	ap_int<2*ROW> ID_reg;

	loop_inputs:
	for(int iter_read = 0; iter_read < size; iter_read++){
		#pragma HLS LOOP_TRIPCOUNT MIN=1000 MAX=1000
		//Read all features into the feature_array.
		loop_stream:
		for(int i = 0; i < N_FEAT_PAD; i++){
			#pragma HLS UNROLL factor=COL_
			feature_stream[i] >> feature_array[i];
		}

		//Probe ROW rows simultanously for mat-vec multplication (result = r encoding dimension).
		//Each row block has Dhv/ROW rows.
		loop_mat_row:
		for(int r = 0; r < Dhv/ROW; r++){
			//Clear the partial encoding buffer when the window starts the new rows.
			loop_clear:
			for(int i = 0; i < ROW; i++){
				#pragma HLS UNROLL factor=ROW_
				encHV_partial[i] = 0;
			}
			//We need to figure out which ID bits should be read.
			//At the beginning of row block r, we read bits of the block r and r+1 (each block has Dhv/ROW bits).
			int cycle = 0;
			int addr1 = r;
			int addr2 = r+1;
			//In the last block, r+1 becomes Dhv/ROW, so we start from 0 (ID bits are stored circular).
			if(addr2 == Dhv/ROW)
				addr2 = 0;
			ID_reg.range(ROW-1, 0) = ID[addr1];
			ID_reg.range(2*ROW-1, ROW) = ID[addr2];

			//Divide each of row blocks into columns (tiles) of COL, i.e., multiply a ROW*COL tile to COL features at a given cycle.
			loop_mat_col:
			for(int c = 0; c < N_FEAT_PAD/COL; c++){
				#pragma HLS PIPELINE

				//Iterate over the rows and columns of the ROW*COL tile to perform matrix-vector multplication.
				loop_tile_row:
				for(int i = 0; i < ROW; i++){
					#pragma HLS UNROLL factor=ROW_
					//In each ID register of 2*ROW bits, bits [0-COL) are for the first row, [1, COL+1) for the second, and so on.
					ap_int<COL> ID_row = ID_reg.range(i+COL-1, i);
					loop_tile_col:
					for(int j = 0; j < COL; j++){
						#pragma HLS UNROLL factor=COL_
						//For column group c, we read features c*COL to (c+1)*COL.
						int feature = feature_array[c*COL + j];
						if(ID_row[j] == 1)
							encHV_partial[i] += feature;
						else
							encHV_partial[i] -= feature;
					}
				}
				//After the first window, we move the window by right.
				//The initial 2*ROW ID block has enough bits for ROW/COL consecutive windows (as each window needs ROW+COL bits, not 2*ROW bits).
				//Otherwise, we update the ID address to get the new required ID bits.
				cycle += 1;
				if(cycle == ROW/COL){
					cycle = 0;
					addr1 = addr1 + 1;
					addr2 = addr2 + 1;
					if(addr1 == Dhv/ROW)
						addr1 = 0;
					if(addr2 == Dhv/ROW)
						addr2 = 0;
					ID_reg.range(ROW-1, 0) = ID[addr1];
					ID_reg.range(2*ROW-1, ROW) = ID[addr2];
				}
				//We have not reached the bound of ROW/COL, so the ID register contains the needed bits.
				//Just shift right by COL, so 'ID_reg.range(i+COL-1, i)' gives the correct ID bits per each row i of the ID block.
				//E.g., in a 4x2 window, in the first cycle we need bits 0-1 for row 1, while in the next cycle we need bits 2-3, so shift by COL=2 is needed.
				else{
					ID_reg = (ID_reg >> COL);
				}
			}
			//Output the ROW generated dimensions for subsequent pipelined search.
			//Note that we use quantized random projection. Otherwise, we will need higher bit-width for classes (and tmp resgiter during dot-product).
			loop_enc_stream:
			for(int i = 0; i < ROW; i++){
				#pragma HLS UNROLL factor=ROW_
				if(encHV_partial[i] >= 0)
					enc_stream[i] << 1;
				else
					enc_stream[i] << -1;
			}
		}
	}
	//Iterating over inputs ends here.
}

/*
 * searchUnit is the major component of our implementation. It runs EPOCH times over the data (EPOCH=1 for inference).
 * In the first epoch, it reads encoding elements ROW by ROW (ROW elements in Div/COL cycles) from the encodeUnit unit, and stores them in the global memory to reuse in later epochs (in case of training).
 * In the remaining epochs, it reads the encoded hypervector from the global memory. Upon reading ROW dimensions, it compares (similarity checking) them with the corresponding dimensions of all classes.
 * After finding the class with highest score during retraining, the model updates in case of misprediction.
 *
 * enc_stream (input): ROW parallel stream of bipolar (+1, -1) dimensions from encoding unit.
 * classHV_gmem (input/output): class hypervectors; output in case of training, and input in case of inference.
 * labels_gmem (input/output): label of data samples; input in case of training, and output in case of inference.
 * encHV_gmem (input/output): interface to write/read encoded hypervectors to/from the DRAM to reuse encoded data.
 * trainScore (output): number of correct predictions in the last epoch of training.
 * train (input): number of training epochs (0 = inference)
 * size (input): number of data samples.
 */

template<int ROW_>
void searchUnit(hls::stream<dt_int2> enc_stream[ROW], int *classHV_gmem, int *labels_gmem, ap_int<512> *encHV_gmem, int *trainScore, int train, int size){

	//Explained previously: to operate on ROW encoding dimensions per cycle.
	dt_int2 encHV_partial[ROW];
	#pragma HLS array_partition variable=encHV_partial

	//To store the dot-product of the classes with the encoding hypervector.
	int dotProductRes[N_CLASS];
	#pragma HLS array_partition variable=dotProductRes

	//For cosine, we need to store 1/|C|_2, which are small fractional numbers. For now we use float, though we may change to ap_fixed.
	float norm2_inv[N_CLASS];

	//During retraining, we will need the encoded hypervector from global memory (as we generated and stored them in the first epoch).
	//I tried replacing encHV_full with a 1-d array (i.e., dt_int2 encHV_full[Dhv] with cyclic partitioning) but latency of search increased 50%.
	//As a result of using 2-d array, there will be some annoying temp variables to read/write data from/to encHV_full within the code.
	ap_int<ROW> encHV_full[Dhv/ROW];
	#pragma HLS array_partition variable=encHV_full

	int EPOCH = (train == 0) ? 1 : train;

	int classHV[N_CLASS][Dhv];
	//We partition each class dimensions into ROW elements to match the ROW generated dimensions.
	#pragma HLS array_partition variable=classHV cyclic factor=ROW_ dim=2

	//Initialize the class hypervectors.
	loop_initClass:
	for(int i = 0; i < N_CLASS; i++){
		for(int dim = 0; dim < Dhv; dim++){
			#pragma HLS PIPELINE
			//For inference, class hypervectors are given.
			if(train == 0)
				classHV[i][dim] = classHV_gmem[i*Dhv + dim];
			//For training, initialize to zero.
			else
				classHV[i][dim] = 0;
		}
	}

	int correct = -1;

	loop_repeat:
	for(int iter_epoch = 0; iter_epoch < EPOCH; iter_epoch++){
		#pragma HLS LOOP_TRIPCOUNT MIN=1 MAX=1

		//Count the number of correct prediction in training epochs.
		correct = 0;

		//At the beginning of each epoch, calculate 1/|C|_2 (we call "1/|C|_2" as norm2).
		loop_norm_1:
		for(int i_class = 0; i_class < N_CLASS; i_class++){
			ap_int<64> total = 0;
			loop_norm_2:
			for(int dim = 0; dim < Dhv; dim++){
				#pragma HLS UNROLL factor=ROW_
				total += classHV[i_class][dim] * classHV[i_class][dim];
			}
			//Total might be 0 before the first round of training, or if some class didn't have any sample,
			//So we use 1/|C|_2 = 0 to make its similarity (H*C*1/|C|_2) score 0 (although similarity checking won't be actually used in the first round of training).
			if(total == 0)
				norm2_inv[i_class] = 0;
			else{
				norm2_inv[i_class] = 1.0 / float(total);
			}
		}

		//cout << "norm2_inv[0]: " << norm2_inv[0] << endl;

		int label;

		loop_inputs:
		for(int iter_read = 0; iter_read < size; iter_read++){
			#pragma HLS LOOP_TRIPCOUNT MIN=1000 MAX=1000

			//For inference we do not need to read the label.
			if(train > 0)
				label = labels_gmem[iter_read];

			//Reset the dotProductRes (score buffer) before each input sample.
			loop_clear:
			for(int i_class = 0; i_class < N_CLASS; i_class++){
				#pragma HLS UNROLL
				dotProductRes[i_class] = 0;
			}
			//In the subsequent training epochs (i.e., retraining), we just reuse the encoding hypervectors generated in the first epoch.
			if(iter_epoch > 0){
				loop_read_encHV:
				for(int i = 0; i < Dhv/512; i++){
					//#pragma HLS PIPELINE
					ap_int<512> enc_512b = encHV_gmem[iter_read*(Dhv/512) + i];
					for(int j = 0; j < 512/ROW; j += 1){
						#pragma HLS UNROLL
						encHV_full[(i*512/ROW) + j] = enc_512b.range(j*ROW+ROW-1, j*ROW);
					}
				}
			}
			//In the first EPOCH, will read Dhv encoding dimensions, ROW by ROW (ROW dimensions per Dhv/COL cycles).
			//i_dim keeps track of the global index of classes (increases by ROW after processing a block of ROW rows).
			loop_outer:
			for(int i_dim = 0; i_dim < Dhv/ROW; i_dim += 1){
				ap_int<ROW> temp_partial = encHV_full[i_dim];
				loop_stream:
				for(int j_sub = 0; j_sub < ROW; j_sub++){
					#pragma HLS UNROLL factor=ROW_
					if(iter_epoch == 0){
						enc_stream[j_sub] >> encHV_partial[j_sub];
					}
					else{
						encHV_partial[j_sub] = temp_partial[j_sub] == 1 ? 1 : -1; //Binary to bipolar conversion.
					}
				}
				 //In the first epoch of TRAINING, initialize the classes, and store the encoded hypervector.
				if(iter_epoch == 0 && train > 0){
					ap_int<ROW> temp_partial;
					loop_init:
					for(int j_sub = 0; j_sub < ROW; j_sub++){
						#pragma HLS UNROLL factor=ROW_
						classHV[label][i_dim*ROW + j_sub] += encHV_partial[j_sub];
						//store the dimensions (in a whole hypervector) and save to global memory for reuse in next epochs.
						temp_partial[j_sub] = encHV_partial[j_sub] == 1 ? 1 : 0; //Bipolar to binary conversion.
					}
					encHV_full[i_dim] = temp_partial;
				}
				//In the next training epochs and/or in inference, calculate the similarity scores.
				else{
					//Multiply the generated ROW encoding dimensions to the corresponding class hypervectors.
					loop_score:
					for(int j_class = 0; j_class < N_CLASS; j_class++){
						//#pragma HLS PIPELINE
						#pragma HLS UNROLL
						loop_inner:
						for(int k_sub = 0; k_sub < ROW; k_sub++){
							#pragma HLS UNROLL factor=ROW_
							//i_dim keeps track of the global index of classes (increases by ROW after processing a block of ROW rows).
							dotProductRes[j_class] += encHV_partial[k_sub] * classHV[j_class][i_dim*ROW + k_sub];
						}
					}
				}
			}
			//Calculate max index (needed in inference and REtraining iterations, but we do it in case of initial training too, to avoid if/else...).
			int maxIndex = -1;
			float maxVal = -(1 << 15);
			loop_max:
			for(int i_class = 0; i_class < N_CLASS; i_class++){
				//Here is the tricky part; I replace H*C/sqrt(|C|_2) by (H*C)^2/|C|_2, while considering the sign of H*C.
				float temp = dotProductRes[i_class]*norm2_inv[i_class];
				float score = temp * dotProductRes[i_class];
				if(dotProductRes[i_class] < 0)
					score = -score;
				if(score > maxVal){
					maxIndex = i_class;
					maxVal = score;
				}
			}

			//If inference, output the index (label) of the class with maximum similarity score.
			if(train == 0){
				labels_gmem[iter_read] = maxIndex;
			}
			//If it is a REtraining epoch, update the correct and mispredicted class.
			else if (iter_epoch > 0){
				if(maxIndex != label){
					loop_update:
					for(int i_sub = 0; i_sub < Dhv/ROW; i_sub++){
						ap_int<ROW> temp_partial = encHV_full[i_sub];
						for(int j = 0; j < ROW; j++){
							#pragma HLS UNROLL
							dt_int2 temp_dim = temp_partial[j] == 1 ? 1 : -1;
							classHV[label][i_sub*ROW + j] += temp_dim;
							classHV[maxIndex][i_sub*ROW + j] -= temp_dim;
						}
					}
				}
				else{
					correct += 1;
				}
			}
			//Processing an input sample ends here. Write the encoded hypervector to global memory in the FIRST epoch, in case of training.
			if(train > 0 && iter_epoch == 0){
				loop_writeEncHV:
				for(int i = 0; i < Dhv/512; i++){
					ap_int<512> enc_512b;
					for(int j = 0; j < 512/ROW; j += 1){
						#pragma HLS UNROLL
						enc_512b.range(j*ROW+ROW-1, j*ROW) = encHV_full[(i*512/ROW) + j];
					}
					encHV_gmem[iter_read*(Dhv/512) + i] = enc_512b;
				}
			}
		}
		//An epoch finishes here.
		if(train > 0 && iter_epoch > 0)
			cout << "Training epoch " << iter_epoch << " accuracy: " << float(correct)/size << endl;
	}

	//At the end of retraining, write back the generated classes.
	if(train > 0){
		loop_writeClasses:
		for(int i = 0; i < N_CLASS; i++){
			for(int j = 0; j < Dhv; j++){
				#pragma HLS PIPELINE
				classHV_gmem[i*Dhv + j] = classHV[i][j];
			}
		}
		trainScore[0] = correct;
	}

}

template<int ROW_, int COL_>
void top(int *input_gmem, int *ID_gmem, int *classHV_gmem, int *labels_gmem, ap_int<512> *encHV_gmem, int *trainScore, int train, int size){

	static hls::stream<int> feature_stream[N_FEAT_PAD];
	#pragma HLS STREAM variable=feature_stream depth=2

	//For now, the encoding stream is integer while we are using bipolar (+1, -1) encoding. Fix it later.
	static hls::stream<dt_int2> enc_stream[ROW];
	#pragma HLS STREAM variable=enc_stream depth=2

	//We have a seed ID of Dhv length, and we partition it to Dhv/ROW pieces of ROW bits as we operate on ROW rows at the same time.
	ap_int<ROW> ID[Dhv/ROW];
	#pragma HLS array_partition variable=ID cyclic factor=4

	//Initialize the seed ID hypervector.
	int offset = 0;
	loop_initID:
	for(int i = 0; i < Dhv/32; i++){
		ap_int<32> ID_int = ID_gmem[i];
		//If ROW is smaller than 32, each IDarray will fill several ID elements.
		if(ROW < 32){
			for(int j = 0; j < 32/ROW; j++){
				ID[i*32/ROW + j] = ID_int.range((j+1)*ROW - 1, j*ROW);
			}
		}//Otherwise, for each ID element, we need to read several IDarray elements.
		else{
			ID[i*32/ROW].range(32*offset + 31, 32*offset) = ID_int;
			offset += 1;
			if(offset == ROW/32)
				offset = 0;
		}
	}

	#pragma HLS dataflow
	inputStream(input_gmem, feature_stream, size);
	encodeUnit<ROW, COL>(feature_stream, ID, enc_stream, size);
	searchUnit<ROW>(enc_stream, classHV_gmem, labels_gmem, encHV_gmem, trainScore, train, size);

}

/*
 * input_gmem (input): input data port; each feature is quantized to an integer.
 * ID_gmem (input): seed ID hypervector, packed to ints.
 * classHV_gmem (input/output): class hypervectors; output in case of training, and input in case of inference.
 * labels_gmem (input/output): label of data samples; input in case of training, and output in case of inference.
 * encHV_gmem (input/output): interface to write/read encoded hypervectors to/from the DRAM to reuse encoded data.
 * trainScore (output): number of correct predictions in the last epoch of training.
 * train (input): number of training epochs (0 = inference)
 * size (input): number of data samples.
 */

extern "C" {
void hd(int *input_gmem, int *ID_gmem, int *classHV_gmem, int *labels_gmem, ap_int<512> *encHV_gmem, int *trainScore, int train, int size){

	#pragma HLS INTERFACE m_axi port=input_gmem   offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=ID_gmem      offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=classHV_gmem offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=labels_gmem  offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi port=encHV_gmem   offset=slave bundle=gmem3
	#pragma HLS INTERFACE m_axi port=trainScore   offset=slave bundle=gmem2

	#pragma HLS INTERFACE s_axilite port=input_gmem   bundle=control
	#pragma HLS INTERFACE s_axilite port=ID_gmem      bundle=control
	#pragma HLS INTERFACE s_axilite port=classHV_gmem bundle=control
	#pragma HLS INTERFACE s_axilite port=labels_gmem  bundle=control
	#pragma HLS INTERFACE s_axilite port=encHV_gmem   bundle=control
	#pragma HLS INTERFACE s_axilite port=trainScore   bundle=control
	#pragma HLS INTERFACE s_axilite port=train        bundle=control
	#pragma HLS INTERFACE s_axilite port=size         bundle=control
	#pragma HLS INTERFACE s_axilite port=return       bundle=control

	top<ROW, COL>(input_gmem, ID_gmem, classHV_gmem, labels_gmem, encHV_gmem, trainScore, train, size);

}
}
