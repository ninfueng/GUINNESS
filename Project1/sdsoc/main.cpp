/*
 * C++ Templete for a Binarized CNN
 *
 *  Created on: 2017/07/01
 *      Author: H. Nakahara
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <bitset>

#include <ap_int.h>

#ifdef __SDSCC__
#include "sds_lib.h"
#else 
#define sds_alloc(x)(malloc(x))
#define sds_free(x)(free(x))
#endif

void BinCNN(
#ifdef __SDSCC__
        int *t_bin_convW,
        int *t_BNFb,
        ap_int<64> t_in_img[32*32],
        int fc_result[10],
        int init
#else 
        int t_bin_convW[74944],
        int t_BNFb[202],
        ap_int<64> t_in_img[32*32],
        int fc_result[10],
        int init
#endif
);

//--------------------------------------------------------------------
// Main Function
//--------------------------------------------------------------------
int main( int argc, char *argv[])
{
    ap_int<64> *t_tmp_img;
    t_tmp_img = (ap_int<64> *)sds_alloc((32*32)*sizeof(ap_int<64>));

    int fc_result[10];
    int rgb, y, x, i, offset;

    // copy input image to f1
    for( y = 0; y < 32; y++){
    	for( x = 0; x < 32; x++){
    		t_tmp_img[y*32+x] = 0;
        }
    }

    // ------------------------------------------------------------------
    printf("load weights\n");
    int *t_bin_convW;
	int *t_BNFb;
	t_bin_convW = (int *)sds_alloc((74944)*sizeof(int));
	t_BNFb   = (int *)sds_alloc((202)*sizeof(int));

	int of, inf, d_value;
	FILE *fp;
	char line[256];

    printf("b0_BNFb.txt\n");
    if( (fp = fopen("b0_BNFb.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    offset = 0;
    for( of = 0; of < 64; of++){
        if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\n");
        sscanf( line, "%d", &d_value);
        t_BNFb[of+offset] = d_value;
    }
    fclose(fp);
    printf("b1_BNFb.txt\n");
    if( (fp = fopen("b1_BNFb.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    offset = 64;
    for( of = 0; of < 64; of++){
        if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\n");
        sscanf( line, "%d", &d_value);
        t_BNFb[of+offset] = d_value;
    }
    fclose(fp);
    printf("b2_BNFb.txt\n");
    if( (fp = fopen("b2_BNFb.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    offset = 128;
    for( of = 0; of < 64; of++){
        if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\n");
        sscanf( line, "%d", &d_value);
        t_BNFb[of+offset] = d_value;
    }
    fclose(fp);
    printf("b3_BNFb.txt\n");
    if( (fp = fopen("b3_BNFb.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    offset = 192;
    for( of = 0; of < 10; of++){
        if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\n");
        sscanf( line, "%d", &d_value);
        t_BNFb[of+offset] = d_value;
    }
    fclose(fp);


    printf("conv0W.txt\n");
    if( (fp = fopen("conv0W.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    offset = 0;
    for( of = 0; of < 64; of++){
        for( inf = 0; inf < 1; inf++){
            for( y = 0; y < 3; y++){
                for( x = 0; x < 3; x++){
                    if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\n"); sscanf( line, "%d", &d_value);
                    t_bin_convW[of*1*3*3+inf*3*3+y*3+x+offset] = d_value;
                }
            }
        }
    }
    fclose(fp);
    printf("conv1W.txt\n");
    if( (fp = fopen("conv1W.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    offset = 576;
    for( of = 0; of < 64; of++){
        for( inf = 0; inf < 64; inf++){
            for( y = 0; y < 3; y++){
                for( x = 0; x < 3; x++){
                    if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\n"); sscanf( line, "%d", &d_value);
                    t_bin_convW[of*64*3*3+inf*3*3+y*3+x+offset] = d_value;
                }
            }
        }
    }
    fclose(fp);
    printf("conv2W.txt\n");
    if( (fp = fopen("conv2W.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    offset = 37440;
    for( of = 0; of < 64; of++){
        for( inf = 0; inf < 64; inf++){
            for( y = 0; y < 3; y++){
                for( x = 0; x < 3; x++){
                    if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\n"); sscanf( line, "%d", &d_value);
                    t_bin_convW[of*64*3*3+inf*3*3+y*3+x+offset] = d_value;
                }
            }
        }
    }
    fclose(fp);
    printf("fc0W.txt\n");
    if( (fp = fopen("fc0W.txt", "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    offset = 74304;
    for( of = 0; of < 10; of++){
        for( inf = 0; inf < 64; inf++){
            if( fgets( line, 256, fp) == NULL)fprintf(stderr,"EMPTY FILE READ\n"); sscanf( line, "%d", &d_value);
            t_bin_convW[of*64+inf+offset] = d_value;
        }
    }
    fclose(fp);


    printf("setup... \n");
	BinCNN( t_bin_convW, t_BNFb, t_tmp_img, fc_result, 1);

    char image_name[256];
    int cnt;

#ifdef __SDSCC__
    sscanf( argv[1], "%s", image_name); // 1st argument: test image (text file)
    sscanf( argv[2], "%d", &cnt); // 2nd argument: # of inferences 
#else 
    sprintf( image_name, "test_img.txt");
    cnt = 1;
#endif


    int pixel;
    printf("LOAD TESTBENCH %s ... ", image_name);
    if( (fp = fopen(image_name, "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    for( y = 0; y < 32; y++){
        for( x = 0; x < 32; x++){
            ap_int<64>tmp = 0;
            for( rgb = 1 - 1; rgb >= 0 ; rgb--){
                if( fgets( line, 256, fp) == NULL)
                    fprintf(stderr,"EMPTY FILE READ\n"); 
                sscanf( line, "%d", &d_value);

                tmp = tmp << 20;

                pixel = d_value;
                tmp |= ( pixel & 0xFFFFF);
            }
            t_tmp_img[ y * 32 + x] = tmp;
        }
    }
    printf("OK\n");
    fclose(fp);

    printf("Inference %d times ... ", cnt);
    for( i = 0; i < cnt; i++){
        BinCNN( t_bin_convW, t_BNFb, t_tmp_img, fc_result, 0);
    }
    printf("OK\n");

    printf("Result\n");
    for( i = 0; i < 10; i++)printf("%5d ", fc_result[i]);
    printf("\n");

    sds_free( t_tmp_img); sds_free( t_bin_convW); sds_free( t_BNFb);

    return 0;
}

// ------------------------------------------------------------------
// END OF PROGRAM
// ------------------------------------------------------------------
