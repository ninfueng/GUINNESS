/*
 * C++ Templete for a Binarized CNN
 *
 *  Created on: 2017/07/01
 *      Author: H. Nakahara
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <math.h>

using namespace std;

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

    // setup socket connection -----------------------------------------
    struct sockaddr_in addr;
    int sock;
    //char buf[32];
    char buf[20000]; // more than 64x64x3(RGB) bytes
    int data;

    char ipadr[512];
    int portnum;

    if( argc != 3){
        printf("USAGE: #./(program).elf [IPADR] [PORTNUM]\n");
        exit(-1);
    }

    sscanf( argv[1], "%s", ipadr);
    sscanf( argv[2], "%d", &portnum);

    printf("[INFO] IPADR=%s PORT=%d\n", ipadr, portnum);

    /* make a socket */
    sock = socket(AF_INET, SOCK_STREAM, 0);
    /* set parameters */
    addr.sin_family = AF_INET;
    addr.sin_port = htons(portnum); //10050
    addr.sin_addr.s_addr = inet_addr(ipadr); //"192.168.2.100"

    /* connect a server (host PC) */
    connect(sock, (struct sockaddr*)&addr, sizeof(addr));

    // main loop -------------------------------------------------------
    while(1){
        // receive data
        // printf("Receive data\n");
        memset(buf, 0, sizeof(buf));
        data = read(sock, buf, sizeof(buf));

        // set pixel
        // printf("Set Pixel");
        for( y = 0; y < 32; y++){
            for( x = 0; x < 32; x++){
                ap_int<64>tmp = 0;
                for( rgb = 0; rgb < 1; rgb++){
                    tmp = tmp << 20;

                    tmp |= ( buf[y * 32 * 3 + x * 3 + rgb] & 0xFFFFF);
                }
                t_tmp_img[ y * 32 + x] = tmp;
            }
        }
        // printf("OK\n");

        // printf("Inference...\n");
        BinCNN( t_bin_convW, t_BNFb, t_tmp_img, fc_result, 0);
        // printf("OK\n");

        // printf("Result\n");
        // for( i = 0; i < 10; i++)printf("%5d ", fc_result[i]);
        // printf("\n");

        // send data to server
        double softmax[10];
        double total_softmax = 0.0;
        double max_val = -9999.0;

        for( i = 0; i < 10; i++){
            if( (double)fc_result[i] > max_val)
            	max_val = fc_result[i];
        }

        for( i = 0; i < 10; i++){
            total_softmax += exp( (double)(fc_result[i]) / max_val);
        }

	for( i = 0; i < 10; i++){
            softmax[i] = (double)exp((double)fc_result[i] / max_val) / total_softmax;
            buf[i] = (char)(softmax[i] * 100.0);

            // printf("i=%d buf=%d softmax=%f\n", i, buf[i], softmax[i]);
        }

        // printf("Send Data");
        write( sock, buf, 10);
    }

    sds_free( t_tmp_img); sds_free( t_bin_convW); sds_free( t_BNFb);
    close(sock);

}

// ------------------------------------------------------------------
// END OF PROGRAM
// ------------------------------------------------------------------
