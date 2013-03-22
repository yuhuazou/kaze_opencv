
//=============================================================================
//
// nldiffusion_functions.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 27/12/2011
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file nldiffusion_functions.cpp
 * @brief Functions for non-linear diffusion applications:
 * 2D Gaussian Derivatives
 * Perona and Malik conductivity equations
 * Perona and Malik evolution
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 */

#include "kaze_nldiffusion_functions.h"

// Namespaces
using namespace std;

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function smoothes an image with a Gaussian kernel
 * @param src Input image
 * @param dst Output image
 * @param ksize_x Kernel size in X-direction (horizontal)
 * @param ksize_y Kernel size in Y-direction (vertical)
 * @param sigma Kernel standard deviation
 */
void Gaussian_2D_Convolution(const cv::Mat &src, cv::Mat &dst, unsigned int ksize_x,
							 unsigned int ksize_y, float sigma)
{
	// Compute an appropriate kernel size according to the specified sigma
	if( sigma > ksize_x || sigma > ksize_y || ksize_x == 0 || ksize_y == 0 )
	{
		ksize_x = ceil(2.0*(1.0 + (sigma-0.8)/(0.3)));
		ksize_y = ksize_x;
	}

	// The kernel size must be and odd number
	if( (ksize_x % 2) == 0 )
	{
		ksize_x += 1;
	}
		
	if( (ksize_y % 2) == 0 )
	{
		ksize_y += 1;
	}

	// Perform the Gaussian Smoothing with border replication
	cv::GaussianBlur(src,dst,cv::Size(ksize_x,ksize_y),sigma,sigma,cv::BORDER_REPLICATE);
	
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes image derivatives with symmetric differences
 * @param src Input image
 * @param dst Output image
 * @param xorder Derivative order in X-direction (horizontal)
 * @param yorder Derivative order in Y-direction (vertical)
 */
void Image_Derivatives_SD(const cv::Mat &src, cv::Mat &dst, unsigned int xorder, unsigned int yorder)
{
   unsigned int norder_x = xorder;
   unsigned int norder_y = yorder;
   int left = 0, right = 0, up = 0, down = 0;
   
   // Initialize the destination image
   dst = cv::Mat::zeros(dst.rows,dst.cols,CV_32F);
   
   // Create an auxiliary image
   cv::Mat aux(dst.rows,dst.cols,CV_32F);
   src.copyTo(aux);
   
   // Firstly compute derivatives in the x-axis (horizontal)
   while( norder_x != 0 )
   {
	   for( int i = 0; i < aux.rows; i++ )
	   {
		   for( int j = 0; j < aux.cols; j++ )
		   {
				left = j-1;
				right = j+1;
				
				// Check the horizontal bounds
				if( left < 0 )
				{
					left = 0;
				}
				
				if( right >= aux.cols)
				{
					right = aux.cols-1;
				}
				
                *(dst.ptr<float>(i)+j) = 0.5*((*(aux.ptr<float>(i)+right))-(*(aux.ptr<float>(i)+left)));
		   }
	   }
	   
	   norder_x--;
	   
	   if( norder_x != 0 )
	   {
		   dst.copyTo(aux);
	   }
   }
   
   // Compute derivatives in the y-axis (vertical)
   while( norder_y != 0 )
   {
	   for( int i = 0; i < aux.cols; i++ )
	   {
		   for( int j = 0; j < aux.rows; j++ )
		   {
				up = j-1;
				down = j+1;
				
				// Check the vertical bounds
				if( up < 0 )
				{
					up = 0;
				}
				
				if( down >= aux.rows)
				{
					down = aux.rows-1;
				}
				
                *(dst.ptr<float>(j)+i) = 0.5*((*(aux.ptr<float>(down)+i))-(*(aux.ptr<float>(up)+i)));
		   }
	   }
	   
	   norder_y--;
	   
	   if( norder_y != 0 )
	   {
		   dst.copyTo(aux);
	   }
   }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes image derivatives with Scharr kernel
 * @param src Input image
 * @param dst Output image
 * @param xorder Derivative order in X-direction (horizontal)
 * @param yorder Derivative order in Y-direction (vertical)
 * @note Scharr operator approximates better rotation invariance than
 * other stencils such as Sobel. See Weickert and Scharr,
 * A Scheme for Coherence-Enhancing Diffusion Filtering with Optimized Rotation Invariance,
 * Journal of Visual Communication and Image Representation 2002
 */
void Image_Derivatives_Scharr(const cv::Mat &src, cv::Mat &dst, unsigned int xorder, unsigned int yorder)
{
   // Compute Scharr filter
   cv::Scharr(src,dst,CV_32F,xorder,yorder,1,0,cv::BORDER_DEFAULT);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes Gaussian image derivatives up to second order
 * @param src Input image
 * @param smooth Smoothed version of the input image
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param Lxy Second order cross image derivative
 * @param Lxx Second order image derivative in X-direction (horizontal)
 * @param Lyy Second order image derivative in Y-direction (vertical)
 * @param ksize_x Kernel size in X-direction (horizontal) for the Gaussian smoothing kernel
 * @param ksize_y Kernel size in Y-direction (vertical) for the Gaussian smoothing kernel
 * @param sigma Standard deviation of the Gaussian kernel
 */
void Compute_Gaussian_2D_Derivatives(const cv::Mat &src, cv::Mat &smooth,cv::Mat &Lx, cv::Mat &Ly,
                                     cv::Mat &Lxy, cv::Mat &Lxx, cv::Mat &Lyy,
                                     unsigned int ksize_x, unsigned int ksize_y, float sigma )
{
	// Firstly, convolve the original image with a Gaussian kernel
	Gaussian_2D_Convolution(src,smooth,ksize_x,ksize_y,sigma);
	
    Image_Derivatives_Scharr(src,Lx,1,0);
    Image_Derivatives_Scharr(src,Ly,0,1);
    Image_Derivatives_Scharr(Lx,Lxx,1,0);
    Image_Derivatives_Scharr(Ly,Lyy,0,1);
    Image_Derivatives_Scharr(Lx,Lxy,0,1);

	// In case we use natural coordinates
	if( use_natural_coordinates == true )
	{
		Lx = Lx*sigma;
		Ly = Ly*sigma;
		Lxx = Lxx*sigma*sigma;
		Lyy = Lyy*sigma*sigma;
		Lxy = Lxy*sigma*sigma;	
	}
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the Perona and Malik conductivity coefficient g1
 * g1 = exp(-|dL|^2/k^2)
 * @param src Input image
 * @param dst Output image
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param k Contrast factor parameter
 */
void PM_G1(const cv::Mat &src, cv::Mat &dst, cv::Mat &Lx, cv::Mat &Ly, float k )
{
    cv::exp(-(Lx.mul(Lx) + Ly.mul(Ly))/(k*k),dst);
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the Perona and Malik conductivity coefficient g2
 * g2 = 1 / (1 + dL^2 / k^2)
 * @param src Input image
 * @param dst Output image
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param k Contrast factor parameter
 */
void PM_G2(const cv::Mat &src, cv::Mat &dst, cv::Mat &Lx, cv::Mat &Ly, float k )
{
    dst = 1./(1. + (Lx.mul(Lx) + Ly.mul(Ly))/(k*k));
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes Weickert conductivity coefficient g3
 * @param src Input image
 * @param dst Output image
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param k Contrast factor parameter
 * @note For more information check the following paper: J. Weickert
 * Applications of nonlinear diffusion in image processing and computer vision,
 * Proceedings of Algorithmy 2000
 */
void Weickert_Diffusivity(const cv::Mat &src, cv::Mat &dst, cv::Mat &Lx, cv::Mat &Ly, float k )
{
    cv::Mat modg;
    cv::pow((Lx.mul(Lx) + Ly.mul(Ly))/(k*k),4,modg);
    cv::exp(-3.315/modg, dst);
    dst = 1.0 - dst;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes a good empirical value for the k contrast factor
 * given an input image, the percentile (0-1), the gradient scale and the number of
 * bins in the histogram
 * @param img Input image
 * @param perc Percentile of the image gradient histogram (0-1)
 * @param gscale Scale for computing the image gradient histogram
 * @param nbins Number of histogram bins
 * @param ksize_x Kernel size in X-direction (horizontal) for the Gaussian smoothing kernel
 * @param ksize_y Kernel size in Y-direction (vertical) for the Gaussian smoothing kernel
 * @return k contrast factor
 */
float Compute_K_Percentile(const cv::Mat &img, float perc, float gscale, unsigned int nbins, unsigned int ksize_x, unsigned int ksize_y)
{
	float kperc = 0.0, modg = 0.0, lx = 0.0, ly = 0.0;
	unsigned int nbin = 0, nelements = 0, nthreshold = 0, k = 0;
    float npoints = 0.0;
	float hmax = 0.0;

    // Create the array for the histogram
    float *hist = new float[nbins];

	// Create the matrices
	cv::Mat gaussian = cv::Mat::zeros(img.rows,img.cols,CV_32F);
	cv::Mat Lx = cv::Mat::zeros(img.rows,img.cols,CV_32F);
	cv::Mat Ly = cv::Mat::zeros(img.rows,img.cols,CV_32F);
	
	// Set the histogram to zero, just in case
	for( unsigned int i = 0; i < nbins; i++ )
	{
		hist[i] = 0.0;
	}

	// Perform the Gaussian convolution
	Gaussian_2D_Convolution(img,gaussian,ksize_x,ksize_y,gscale);
			
	// Compute the Gaussian derivatives Lx and Ly
    Image_Derivatives_Scharr(gaussian,Lx,1,0);
    Image_Derivatives_Scharr(gaussian,Ly,0,1);
	
	// Skip the borders for computing the histogram
	for( int i = 1; i < gaussian.rows-1; i++ )
	{
		for( int j = 1; j < gaussian.cols-1; j++ )
		{
            lx = *(Lx.ptr<float>(i)+j);
            ly = *(Ly.ptr<float>(i)+j);
			modg = sqrt(lx*lx + ly*ly);
	
			// Get the maximum
			if( modg > hmax )
			{
				hmax = modg;
			}
		}
	}

	// Skip the borders for computing the histogram
	for( int i = 1; i < gaussian.rows-1; i++ )
	{
		for( int j = 1; j < gaussian.cols-1; j++ )
		{
            lx = *(Lx.ptr<float>(i)+j);
            ly = *(Ly.ptr<float>(i)+j);
			modg = sqrt(lx*lx + ly*ly);

			// Find the correspondent bin
			if( modg != 0.0 )
			{
				nbin = floor(nbins*(modg/hmax));

                if( nbin == nbins )
                {
                    nbin--;
                }

				hist[nbin]++;
				npoints++;
			}
		}
	}
	
	// Now find the perc of the histogram percentile
	nthreshold = (unsigned int)(npoints*perc);
	
	for( k = 0; nelements < nthreshold && k < nbins; k++)
	{
		nelements = nelements + hist[k];
	}
	
	if( nelements < nthreshold )
	{
		kperc = 0.03;
	}
	else
	{
		kperc = hmax*((float)(k)/(float)nbins);	
	}
	
    delete hist;
	return kperc;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes Scharr image derivatives
 * @param src Input image
 * @param dst Output image
 * @param xorder Derivative order in X-direction (horizontal)
 * @param yorder Derivative order in Y-direction (vertical)
 * @param scale Scale factor or derivative size
 */
void Compute_Scharr_Derivatives(const cv::Mat &src, cv::Mat &dst, int xorder, int yorder, int scale )
{
   int a_i = 0, b_i = 0, c_i = 0, d_i = 0, e_i = 0, f_i = 0;
   int a_j = 0, b_j = 0, c_j = 0, d_j = 0, e_j = 0, f_j = 0;
   float sum_pos = 0.0, sum_neg = 0.0, w = 0.0, norm = 0.0;

   // Values for the Scharr kernel
   w = 10.0/3.0;
   norm = 1.0/(2.0*scale*(w+2.0));

   // Horizontal derivative
   // Lx = (1/(2*scale))*(L(i,j+scale)-L(i,j-scale))
   if( xorder == 1 && yorder == 0 )
   {
	   for( int i = 0; i < src.rows; i++ )
	   {
		  for( int j = 0; j < src.cols; j++ )
		  {
               sum_pos = sum_neg = 0.0;
			   a_i = i-scale;	a_j = j-scale;
			   b_i = i-scale;	b_j = j+scale;
			   c_i = i;			c_j = j-scale;
			   d_i = i;			d_j = j+scale;
			   e_i = i+scale;	e_j = j-scale;
			   f_i = i+scale;	f_j = j+scale;
	
			   // Check the bounds of the mask
			   // i -> rows, j -> cols
			   if( a_i < 0 )			a_i = 0;
			   if( a_j < 0 )			a_j = 0;
				
			   if( b_i < 0)				b_i = 0;
			   if( b_j >= src.cols )	b_j = src.cols-1;
				
			   if( c_j < 0 )			c_j = 0;
			   if( d_j >= src.cols)		d_j = src.cols-1;
				
			   if( e_i >= src.rows )	e_i = src.rows-1;
			   if( e_j < 0 )			e_j = 0;
				
			   if( f_i >= src.rows )	f_i = src.rows-1;
			   if( f_j >= src.cols )	f_j = src.cols-1;
			   
               sum_pos += w*(*(src.ptr<float>(d_i)+d_j));
               sum_pos += 1.0*(*(src.ptr<float>(b_i)+b_j));
               sum_pos += 1.0*(*(src.ptr<float>(f_i)+f_j));

               sum_neg += w*(*(src.ptr<float>(c_i)+c_j));
               sum_neg += 1.0*(*(src.ptr<float>(a_i)+a_j));
               sum_neg += 1.0*(*(src.ptr<float>(e_i)+e_j));

               *(dst.ptr<float>(i)+j) = norm*(sum_pos - sum_neg);
		  }
	   }
   }
   // Vertical derivative
   // Ly = (1/(2*scale))*(L(i+scale,j)-L(i-scale,j))
   else if( xorder == 0 && yorder == 1 )
   {
	   for( int j = 0; j < src.cols; j++ )
	   {
		  for( int i = 0; i < src.rows; i++ )
		  {
               sum_pos = sum_neg = 0.0;
			   a_i = i-scale;	a_j = j-scale;
			   b_i = i+scale;	b_j = j-scale;
			   c_i = i-scale;	c_j = j;
			   d_i = i+scale;	d_j = j;
			   e_i = i-scale;	e_j = j+scale;
			   f_i = i+scale;	f_j = j+scale;

			   // Check the bounds of the mask
			   if( a_i < 0 )			a_i = 0;
			   if( a_j < 0 )			a_j = 0;
				
			   if( b_i >= src.rows  )	b_i = src.rows-1;
			   if( b_j < 0 )			b_j = 0;
				
			   if( c_i < 0 )			c_i = 0;
			   if( d_i >= src.rows )	d_i = src.rows-1;
				
			   if( e_i < 0 )			e_i = 0;
			   if( e_j >= src.cols )	e_j = src.cols-1;
				
			   if( f_i >= src.rows )	f_i = src.rows-1;
			   if( f_j >= src.cols )	f_j = src.cols-1;

               sum_pos += w*(*(src.ptr<float>(d_i)+d_j));
               sum_pos += 1.0*(*(src.ptr<float>(b_i)+b_j));
               sum_pos += 1.0*(*(src.ptr<float>(f_i)+f_j));

               sum_neg += w*(*(src.ptr<float>(c_i)+c_j));
               sum_neg += 1.0*(*(src.ptr<float>(a_i)+a_j));
               sum_neg += 1.0*(*(src.ptr<float>(e_i)+e_j));

               *(dst.ptr<float>(i)+j) = norm*(sum_pos - sum_neg);
		  }
	   }
   }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function performs a scalar non-linear diffusion step
 * @param Ld2 Output image in the evolution
 * @param c Conductivity image
 * @param Ld1 Previous image in the evolution
 * @param stepsize The step size in time units
 * @note Forward Euler Scheme 3x3 stencil
 * The function c is a scalar value that depends on the gradient norm
 * dL_by_ds = d(c dL_by_dx)_by_dx + d(c dL_by_dy)_by_dy
 */
void NLD_Step_Scalar(cv::Mat &Ld2, const cv::Mat &Ld1, const cv::Mat &c, float stepsize)
{
	// Auxiliary variables
	float xpos = 0.0, xneg = 0.0, ypos = 0.0, yneg = 0.0;
	int ipos = 0, ineg = 0, jpos = 0, jneg = 0;

    for( int i = 0; i < Ld2.rows; i++ )
	{
        for( int j = 0; j < Ld2.cols; j++ )
		{
			ineg = i-1;
			ipos = i+1;
			jneg = j-1;
			jpos = j+1;
			
			if( ineg < 0 )	ineg = 0;
            if( ipos >= Ld2.rows )	ipos = Ld2.rows-1;
			if( jneg < 0 )	jneg = 0;
            if( jpos >= Ld2.cols )	jpos = Ld2.cols-1;
			
            xpos = ((*(c.ptr<float>(i)+j))+(*(c.ptr<float>(i)+jpos)))*((*(Ld1.ptr<float>(i)+jpos))-(*(Ld1.ptr<float>(i)+j)));
            xneg = ((*(c.ptr<float>(i)+jneg))+(*(c.ptr<float>(i)+j)))*((*(Ld1.ptr<float>(i)+j))-(*(Ld1.ptr<float>(i)+jneg)));

            ypos = ((*(c.ptr<float>(i)+j))+(*(c.ptr<float>(ipos)+j)))*((*(Ld1.ptr<float>(ipos)+j))-(*(Ld1.ptr<float>(i)+j)));
            yneg = ((*(c.ptr<float>(ineg)+j))+(*(c.ptr<float>(i)+j)))*((*(Ld1.ptr<float>(i)+j))-(*(Ld1.ptr<float>(ineg)+j)));

            *(Ld2.ptr<float>(i)+j) =  *(Ld1.ptr<float>(i)+j) + 0.5*stepsize*(xpos-xneg+ypos-yneg);
		}
	}
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function checks if a given pixel is a maximum in a local neighbourhood
 * @param img Input image where we will perform the maximum search
 * @param dsize Half size of the neighbourhood
 * @param value Response value at (x,y) position
 * @param row Image row coordinate
 * @param col Image column coordinate
 * @param same_img Flag to indicate if the image value at (x,y) is in the input image
 * @return 1->is maximum, 0->otherwise
 */
bool Check_Maximum_Neighbourhood(cv::Mat &img, int dsize, float value, int row, int col, bool same_img )
{
	bool response = true;

	for( int i = row-dsize; i <= row+dsize; i++ )
	{
		for( int j = col-dsize; j <= col+dsize; j++ )
		{
			if( i >= 0 && i < img.rows && j >= 0 && j < img.cols )
            {
				if( same_img == true )
				{
					if( i != row || j != col )
					{
                        if( (*(img.ptr<float>(i)+j)) > value )
						{
							response = false;
							return response;
						}
					}		
				}
				else
				{
                    if( (*(img.ptr<float>(i)+j)) > value )
					{
						response = false;
						return response;
					}
				}
			}
		}
	}
	
	return response;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function checks if a given pixel is a minimum in a local neighbourhood
 * @param img Input image where we will perform the minimum search
 * @param dsize Half size of the neighbourhood
 * @param value Response value at (x,y) position
 * @param row Image row coordinate
 * @param col Image column coordinate
 * @param same_img Flag to indicate if the image value at (x,y) is in the input image
 * @return 1->is a minimum, 0->otherwise
 */
bool Check_Minimum_Neighbourhood(cv::Mat &img, int dsize, float value, int row, int col, bool same_img )
{
	bool response = true;

	for( int i = row-dsize; i <= row+dsize; i++ )
	{
		for( int j = col-dsize; j <= col+dsize; j++ )
		{
			if( i >= 0 && i < img.rows && j >= 0 && j < img.cols )
            {
				if( same_img == true )
				{
					if( i != row || j != col )
					{
                        if( (*(img.ptr<float>(i)+j)) <= value )
						{
							response = false;
							return response;
						}
					}		
				}
				else
				{
                    if( (*(img.ptr<float>(i)+j)) <= value )
					{
						response = false;
						return response;
					}
				}
			}
		}
	}
	
	return response;
}
