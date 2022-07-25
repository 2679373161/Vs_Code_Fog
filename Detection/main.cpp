#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/photo/photo.hpp"
#include "opencv2/core.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>  
#include <list>  
#include <fstream>
#include "Ployfit.h"
#include <io.h>
#include "windows.h"
#include <tchar.h>
#include <fstream>
#include<time.h>
#include<stdio.h>
#include <stack>
#include <stdlib.h>
using namespace cv;
using namespace std;
using namespace czy;
//白底参数
#define white_AeraremoveScratch_1      7 //14
#define white_AeraremoveScratch_2      8 //22
#define white_AeraremoveScratch_3      9 // 22
#define white_sideLightStddev          5.2
#define white_xyDirect_removeScratchFlag_1     3
#define white_xyDirect_removeScratchFlag_2     5
//黑底参数
#define black_KernelSize         24
#define black_GrayDifference     24
#define black_Area_min           300
#define black_Area_max           50000
#define black_LongShortlowerth       6
#define black_lengthMinHighLimit     40
#define black_lengthMaxLowLimit      60
#define black_AeraremoveScratch_1    0.877
#define black_AeraremoveScratch_2    1.31
#define black_LenthMin_HighLimit     18
#define black_meanBlackDiffLowLimit            4
#define black_xyDirect_removeScratchFlag_1     3
#define black_xyDirect_removeScratchFlag_2     5
//灰底参数
#define gray_KernelSize              17
#define gray_GrayDifference          3
#define grayColor_GrayDifference     17
#define gray_Area_min     30
#define gray_Area_max     50000
#define gray_LongShortlowerth     8
#define gray_LengthMinHighLimit   60
#define gray_LengthMaxLowLimit    40
#define gray_AeraremoveScratch_1     1.05
#define gray_AeraremoveScratch_2     1.3
#define gray_LenthMin_HighLimit      21
#define gray_LenthMax_LowLimit         92
#define grayColor_LenthMin_HighLimit_1 15
#define grayColor_LenthMax_LowLimit_1     95
#define grayColor_LenthMin_HighLimit_2    30
#define grayColor_LenthMax_LowLimit_2     500
#define gray_xyDirect_removeScratchFlag_1 3
#define gray_xyDirect_removeScratchFlag_2 5
void getSubdirs(std::string path, std::vector<std::string>& files);
Mat toushi_white(Mat image, Mat M, int border, int length, int width);
Mat Gabor6(Mat img_1);
Mat Gabor7(Mat img_1);
Mat Gabor9(Mat img_1);
Mat Gabor5(Mat img_1);
void RemoveSmallRegion(Mat &Src, Mat &Dst, int AreaLimit, int CheckMode, int NeihborMode);
void adaptiveThresholdCustom(const cv::Mat& src, cv::Mat& dst, double maxValue, int method, int type, int blockSize, double delta, double ratio, bool filledsSrategy);
Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB);
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_lightleak, Mat *Mwhite, Mat *Mblack, Mat *Mlightleak, Mat *M_white_abshow);
Mat gamma(Mat src, double g);
bool lack_line_black(Mat white_yiwu, Mat ceguang, Mat *mresult, string *causecolor, string name);
bool lack_line(Mat white_yiwu, Mat ceguang,Mat Mask_MY, Mat* mresult, string *causecolor, string name);
bool lack_line_gray(Mat white_yiwu, Mat ceguang, Mat* mresult, string *causecolor, string name);
bool lack_line_color(Mat white_yiwu, Mat ceguang, Mat *mresult, string *causecolor);
bool Stripe(Mat gray, Mat* mresult, string* causecolor);
bool lack_line_red(Mat src_red, Mat ceguang, Mat* mresult, string* causecolor);
bool edgelackline(Mat src_red, Mat ceguang, Mat* mresult, string* causecolor, string name);
bool spurtcode(Mat white_yiwu, Mat ceguang, Mat* mresult, string* causecolor);
bool lack_line_rgb(Mat src_red, Mat ceguang, Mat* mresult, string* causecolor, string name);
bool compareContourSizes(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2);
bool mainCamPersTransMatCalwhite(InputArray _src, int border_white, Mat* Mwhite);
void convexSetPretreatment(Mat& src);
bool lack_line_rgb_My(Mat src_red, Mat ceguang, Mat src_red_old , Mat Shield_MY_IN, Mat* mresult, string* causecolor, string name);

Mat convertTo3Channels(const Mat& binImg);

bool sortContoursLength(vector<cv::Point> contour1, vector<cv::Point> contour2);
double CalcMHWScore(vector<int> scores);

Mat f_remove_mark(Mat src, Mat ceguang);



// 对比度
cv::Mat Contrast(cv::Mat src, float percent, int thresh)
{
	//float alpha = percent / 100.f;
	float alpha = max(-1.f, min(1.f, percent));
	cv::Mat temp = src.clone();
	int row = src.rows;
	int col = src.cols;

	for (int i = 0; i < row; ++i)
	{
		uchar* t = temp.ptr<uchar>(i);
		uchar* s = src.ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			uchar b = s[j];

			int newb, newg, newr;
			if (alpha == 1)
			{

				t[j] = b > thresh ? 255 : 0;
				continue;
			}
			else if (alpha >= 0)
			{
				newb = static_cast<int>(thresh + (b - thresh) / (1 - alpha));
			}
			else {

				newb = static_cast<int>(thresh + (b - thresh) * (1 + alpha));

			}

			newb = max(0, min(255, newb));

			t[j] = static_cast<uchar>(newb);
		}
	}
	return temp;
}

//比较函数对象
bool compareContourAreas(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}

bool compareContourSizes(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2)
{
	return contour1.size() > contour2.size();
}

//string rootPath = "D:\\test\\FOG\\SX\\Standard_sample\\膜划伤\\水平膜划伤2\\1245SX";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
//string rootPath = "D:\\Test_result\\RJ\\station\\20220514182\\24SX";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
//string rootPath = "D:\\test\\FOG\\SX\\Standard_sample\\彩底图_策略2检出的较淡少线\\69";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
string rootPath = "D:\\Test_result\\V_SX\\34";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
//string rootPath = "D:\\Test_result\\V_LP\\_2\\station_白底马克笔\\20220514182\\504SX";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_

//string rootPath = "D:\\test\\FOG\\SX\\少线漏检0712\\142_PB";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
//string rootPath = "D:\\test\\FOG\\SX\\Marke\\7_4_sample\\6XYSX";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
string rootPath1 = "D:\\test\\FOG\\SX\\Marke";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
ofstream  csvFile("D:\\test\\检测结果.csv");//保存结果路径

string cyk;
string Blankimage;
string outname = "\\死灯特征3.csv";
string Sample_number = "HF-ACA03-LP";
string Sample_number1;
FILE* pOutFile;                               //输出文件指针
FILE* pOutFile1;                              //输出文件指针
string path1;
double timeLength;
int sampleNum;
bool Ext_Result_Left_Right;
DWORD t1, t2;

void getSubdirs(std::string path, std::vector<std::string>& files)
{
	long long hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))//是目录
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)//
				{
					getSubdirs(p.assign(path).append("\\").append(fileinfo.name), files);
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}
			}
			else//是文件不是目录
			{
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				if (std::find(files.begin(), files.end(), p.assign(path)) == files.end())
					files.push_back(p.assign(path));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

vector<string> split(string str, char del) {
	stringstream ss(str);
	string temp;
	vector<string> ret;
	while (getline(ss, temp, del)) {
		ret.push_back(temp);
	}
	return ret;
}



int main()
{
	//vector<vector<string>> results;
	vector<string> filenames;//用来存储文件名	
	//cyk.append(rootPath1).append(outname);

	/*	Sample_number1=Sample_number.data;*/
	/*const char* cyk1 = cyk.data();

	if ((fopen_s(&pOutFile, cyk1, "w")) != NULL)
	{
		printf("can not write on file");
		exit(0);
	}*/

	getSubdirs(rootPath, filenames);//获得根目录下所有图片路径,根目录下所有文件

	
	
	if (filenames.size() == 0)
		filenames.push_back(rootPath);
	for (int g = 0; g < filenames.size(); g++)
	{
		string Path = filenames[g] + "\\";
		cout << Path << endl;
		vector<string> res = split(Path, '\\');        //字符串切片，按指定符号分割，返回
		//fprintf(pOutFile, "%s,", res[res.size()-1].c_str());
		string Imgname = res[res.size() - 1].c_str();
		Mat src_white = cv::imread(Path + "WhiteROI.bmp", -1);
		Mat src_ceguang = cv::imread(Path + "CeguangROI.bmp", -1);
		Mat src_black = cv::imread(Path + "BlackROI.bmp", -1);
		Mat src_gray = cv::imread(Path + "GrayROI.bmp", -1);
		/*Mat src_red = cv::imread(Path + "src_color0418.bmp", -1);*/
		Mat src_rgb = cv::imread(Path + "RgbROI.bmp", -1);
		Mat src_ShieldMask_MY = cv::imread(Path + "ShieldMask.bmp", -1);

		Mat M_white;	      //存储视透变换的矩阵
		Mat M_black;
		Mat M_louguang;
		Mat M_white_abshow;
		Mat Mresult_1_white; //检测结果图
		string causeColor_1_white = "";


		//mainCamPersTransMatCalwhite(src_white, 0, &M_white);//黑白相机ROI变换矩阵	判断是否显示异常，并得出视透变换3*3矩阵
		//Mat white = toushi_white(src_white, M_white, -1, 3000, 1500);
		//Mat ceguang = toushi_white(src_ceguang, M_white, -1, 3000, 1500);	
		//Mat black = toushi_white(src_black, M_white, -1, 3000, 1500);	
		//Mat gray_1 = toushi_white(src_gray, M_white, -1, 3000, 1500);
		//Mat rgb = toushi_white(src_rgb, M_white, -1, 3000, 1500);

		//Mat red = toushi_white(src_red, M_white_2, -5, 3000, 1500);
		//cvtColor(white, white, CV_BGR2GRAY);
		//cvtColor(ceguang, ceguang, CV_BGR2GRAY);

		bool flagW = false;
		bool flagR = false;
		bool flagB = false;
		bool flagG = false;
		bool flagS = false;
		Mat whitefilter = Gabor7(src_white);
		//Mat whitefilter = src_white.clone();
		double meanAll = mean(src_white)[0];
		Mat black_Gabor2 = Gabor5(src_black);
		Mat mainfilter = Gabor7(src_rgb);           //滤波去除水平和竖直方向的纹理
		//Mat mainfilter = src_rgb.clone();           //滤波去除水平和竖直方向的纹理
		Mat mainfilter1 = Gabor7(src_gray);        //滤波去除水平和竖直方向的纹理
		//Mat mainfilter2 = Gabor6(red);

		/*Mat My_Need = cv::imread("save.bmp", -1);
		Mat th1_Temp;
		threshold(My_Need, th1_Temp, 1, 255, CV_THRESH_BINARY);
		bitwise_and(~th1_Temp, whitefilter, th1_Temp);
		th1_Temp = My_Need + th1_Temp;*/

		//Mat image[3];
		//split(whitefilter, image);
		//Mat img1 = image[0];
		//Mat img2 = image[1];
		//Mat img3 = image[2];
		//black_Gabor2(Rect(black_Gabor2.cols , black_Gabor2.rows , 0, 0)) = uchar(0);//去除易撕贴部分域

		t1 = GetTickCount(); //它返回从操作系统启动到当前所经过的毫秒数，常常用来判断某个方法执行的时间
		Mat lacklineEenhance1 = gamma(mainfilter1, 0.5);//灰底用幂后的图像 0.8

		//Mat End = f_remove_mark(whitefilter, src_ceguang);

		

		flagW = lack_line(whitefilter, src_ceguang, src_ShieldMask_MY, &Mresult_1_white, &causeColor_1_white,Imgname);
		//flagR = lack_line_rgb(mainfilter, ceguang, &Mresult_1_white, &causeColor_1_white, Imgname);//加入少线缺陷判断
		//lack_line_rgb_My(mainfilter, src_ceguang,src_rgb ,src_ShieldMask_MY , &Mresult_1_white, &causeColor_1_white, Imgname);
		//flagB = lack_line_black(black_Gabor2, ceguang, &Mresult_1_white, &causeColor_1_white, Imgname);
		
		//flagG = lack_line_gray(lacklineEenhance1, src_ceguang, &Mresult_1_white, &causeColor_1_white, Imgname);// lacklineEenhance
		//flagS = edgelackline(whitefilter, ceguang, &Mresult_1_white, &causeColor_1_white, Imgname);
		/*flag = Stripe(mainfilter1, &Mresult_1_white, &causeColor_1_white);*/

		Mat color_lack_line;
		//cvtColor(gray_color, color_lack_line, CV_BGR2GRAY);
		//Mat color_lack = Gabor5(color_lack_line);//彩色相机灰底滤波

		//string namec = res[res.size() - 1].c_str();
		//if (flag == true) {
		//	imwrite("d:\\result\\" + namec + ".bmp", Mresult_1_white);
		//}

		//flag = lack_line_red(mainfilter2, ceguang_C1,&Mresult_1_white, &causeColor_1_white);//红阶少线
		//flag = lack_line_color(lacklineEenhance1, ceguang, &Mresult_1_white, &causeColor_1_white);// lacklineEenhance

		//vector<string> result;
		//result.push_back(filenames[g]);
		//if (causeColor_1_white != "")
		//	result.push_back(causeColor_1_white); 
		//else
		//	result.push_back("良品");  
		//results.push_back(result);
		//cout << result[0] << ":            "<<result[1] << endl;
		/*fprintf(pOutFile, "%s\n", result[1].c_str());*/

		if (flagW == true) {
			cout << "白底：少线  ";
		}
		else
			cout << "白底：OK  ";
		if (flagR == true) {
			cout << "彩底：少线  ";
		}
		else
			cout << "彩底：OK  ";
		if (flagB == true) {
			cout << "黑底：少线  ";
		}
		else
			cout << "黑底：OK  ";
		if (flagG == true) {
			cout << "灰底：少线  " << endl;
		}
		else
			cout << "灰底：OK  " << endl;
		if (flagS == true) {
			cout << "白底：边缘少线  " << endl;
		}
		else
			cout << "白底：边缘OK  " << endl;

		t2 = GetTickCount();
		double dur = (double)(t2 - t1);
		cout << "耗时" << dur << endl;
		//timeLength += dur;

	}
	//csvFile << "样本路径" << "," << "检测结果" << "\n";
	//for (int i = 0; i < results.size(); i++)
	//{
	//	csvFile << results[i][0] << "," << results[i][1] << "\n";
	//}
	//cout << timeLength / filenames.size() << endl;
	//csvFile << "平均时间" << timeLength / filenames.size() << endl;

	/*csvFile.flush();
	csvFile.close();*/
	system("pause");
	return 0;
}

/*=========================================================
* 函 数 名: lack_line_color
* 功能描述: 少线缺陷判断(彩色)
* 函数输入：主相机白底图像和主相机拍摄侧光图像
* 备注说明：2020年12月26日修改
=========================================================*/
bool lack_line_color(Mat white_yiwu, Mat ceguang, Mat *mresult, string *causecolor)
{
	bool result = false;
	Mat img_gray = white_yiwu.clone();
	//imwrite("D://img_gray.bmp",img_gray);
	Mat gray_ceguang = ceguang.clone();
	Mat imageBinary;//分割二值图
	Mat imageBinary2;//分割二值图
	Mat imageBinary1;
	//统计同一列中独立像素点个数
	vector<int> X_list;
	vector<int> Y_list;
	vector<int> count_x;
	vector<int> count_y;
	count_x.resize(img_gray.rows);
	count_y.resize(img_gray.cols);
	medianBlur(img_gray, img_gray, 3);
	//adaptiveThreshold(img_gray, imageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 3);//lackline_bolckSize, lackline_delta
	adaptiveThresholdCustom(img_gray, imageBinary1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, -3, 1, 0.5);//21 5  LQY:17 3
	adaptiveThreshold(img_gray, imageBinary2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 55, 3);
	imageBinary = ~imageBinary1;
	//Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));//膨胀操作结构元素
	//dilate(imageBinary, imageBinary, element);
	//imageBinary(Rect(imageBinary.cols - 300, imageBinary.rows - 300, 300, 300)) = uchar(0);//去除易撕贴部分
	//imageBinary(Rect(imageBinary.cols - 180, imageBinary.rows - 200, 180, 200)) = uchar(0);//去除易撕贴部分域
//        imageBinary(Rect(imageBinary.cols - 220, imageBinary.rows - 210, 220, 40)) = uchar(0);//去除易撕贴部分域
//        imageBinary(Rect(imageBinary.cols - 220, imageBinary.rows - 210, 90, 210)) = uchar(0);//去除易撕贴部分域
	//imageBinary(Rect(imageBinary.cols - 220, imageBinary.rows - 340, 220, 340)) = uchar(0);//去除易撕贴部分域
	//imageBinary(Rect(0, 0, imageBinary.cols - 1, 15)) = uchar(0);                          //去除四个边界的影响上边 10
	//imageBinary(Rect(0, imageBinary.rows - 15, imageBinary.cols - 1, 15)) = uchar(0);      //下边 12
	//imageBinary(Rect(0, 0, 20, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	//imageBinary(Rect(imageBinary.cols - 18, 0, 18, imageBinary.rows - 1)) = uchar(0);      //右边15-18
	imageBinary(Rect(imageBinary.cols - 250, 0, 250, 250)) = uchar(0);//去除易撕贴部分域
	imageBinary(Rect(0, 0, imageBinary.cols - 1, 25)) = uchar(0);                          //去除四个边界的影响上边 10
	imageBinary(Rect(0, imageBinary.rows - 20, imageBinary.cols - 1, 20)) = uchar(0);      //下边 12
	imageBinary(Rect(0, 0, 60, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	imageBinary(Rect(imageBinary.cols - 80, 0, 80, imageBinary.rows - 1)) = uchar(0);      //右边15-18
	imageBinary(Rect(imageBinary.cols - 130, 525, 130, 440)) = uchar(0);

	//imageBinary2(Rect(imageBinary2.cols - 220, imageBinary2.rows - 340, 220, 340)) = uchar(0);//去除易撕贴部分域
	//imageBinary2(Rect(0, 0, imageBinary2.cols - 1, 15)) = uchar(0);                          //去除四个边界的影响上边 10
	//imageBinary2(Rect(0, imageBinary2.rows - 15, imageBinary2.cols - 1, 15)) = uchar(0);      //下边 12
	//imageBinary2(Rect(0, 0, 20, imageBinary2.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	//imageBinary2(Rect(imageBinary2.cols - 18, 0, 18, imageBinary2.rows - 1)) = uchar(0);      //右边15-18

	imageBinary2(Rect(imageBinary.cols - 250, 0, 250, 250)) = uchar(0);//去除易撕贴部分域
	imageBinary2(Rect(0, 0, imageBinary.cols - 1, 25)) = uchar(0);                          //去除四个边界的影响上边 10
	imageBinary2(Rect(0, imageBinary.rows - 20, imageBinary.cols - 1, 20)) = uchar(0);      //下边 12
	imageBinary2(Rect(0, 0, 60, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	imageBinary2(Rect(imageBinary.cols - 80, 0, 80, imageBinary.rows - 1)) = uchar(0);      //右边15-18
	imageBinary2(Rect(imageBinary.cols - 130, 525, 130, 440)) = uchar(0);

	//imageBinary(Rect(imageBinary.cols - 231, 0, 230, 320)) = uchar(0);
	//imwrite("D:\\1imageBin.bmp", imageBinary);
	vector<vector<Point>> contours_lackline;
	findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	if (contours_lackline.size() > 1100 && contours_lackline.size() < 2500)//1500  1000
	{
		Mat structure_element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(imageBinary, imageBinary, structure_element2);    //将模板腐蚀一下,为了去除边界影响,否则相与过后会有白边
		contours_lackline.clear();
		findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	}
	else if (contours_lackline.size() >= 2500)
		return false;

	vector<vector<Point>> contours_lackBlackLine;
	findContours(imageBinary2, contours_lackBlackLine, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	vector<Rect> boundRect_lackLine_new(contours_lackBlackLine.size());
	{
		for (vector<int>::size_type i = 0; i < contours_lackBlackLine.size(); i++)
		{

			double area = contourArea(contours_lackBlackLine[i]);

			RotatedRect rect = minAreaRect(contours_lackBlackLine[i]);  //包覆轮廓的最小斜矩形
			Point p = rect.center;
			//按图片坐标系为准，左上角原点,上侧Y轴，左侧X轴
			X_list.push_back(p.y);
			Y_list.push_back(p.x);

			if (area > gray_Area_min && area < gray_Area_max)
			{
				Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
				drawContours(temp_mask, contours_lackBlackLine, i, 255, FILLED, 8);
				boundRect_lackLine_new[i] = boundingRect(Mat(contours_lackBlackLine[i]));
				float w = boundRect_lackLine_new[i].width;
				float h = boundRect_lackLine_new[i].height;
				int X_1 = boundRect_lackLine_new[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect_lackLine_new[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect_lackLine_new[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect_lackLine_new[i].br().y;//矩形右下角Y坐标值
				Moments m = moments(contours_lackBlackLine[i]);
				int x_point = int(m.m10 / m.m00);
				int y_point = int(m.m01 / m.m00);
				if (x_point < 0 || x_point >= imageBinary.cols || y_point < 0 || y_point >= imageBinary.rows)
				{
					continue;
				}
				double longShortRatio = max(h / w, w / h);
				double 	longShortlowerth = gray_LongShortlowerth;//11 4
				int shortHigherth = gray_LengthMinHighLimit;
				//if (longShortRatio > longShortlowerth && min(w, h) < shortHigherth && max(w, h) > 40)//长宽比，长度，宽度限制
				if (longShortRatio > longShortlowerth && min(w, h) < shortHigherth && max(w, h) > gray_LengthMaxLowLimit)//长宽比，长度，宽度限制
				{
					int border = 3;//选定框边界宽度
					int x_lt = X_1 - border;
					//越界保护
					if (x_lt < 0)
					{
						x_lt = 0;
					}
					int y_lt = Y_1 - border;
					if (y_lt < 0)
					{
						y_lt = 0;
					}
					int x_rt = X_2 + border;
					if (x_rt > img_gray.size[1] - 1)
					{
						x_rt = img_gray.size[1] - 1;
					}
					int y_rt = Y_2 + border;
					if (y_rt > img_gray.size[0] - 1)
					{
						y_rt = img_gray.size[0] - 1;
					}
					//侧光图排除明显划痕
					Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 5, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
					Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
					double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
					double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
					double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                        //排除侧光图贴膜划痕的参数
					double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
					double removeScratchFlag;
					//计算方差
					cv::Mat meanGray;
					cv::Mat stdDev;
					cv::meanStdDev(sidelightSuspect, meanGray, stdDev);
					double sideLightStddev = stdDev.at<double>(0, 0);
					if (removeScratchArea > 10000)
						removeScratchFlag = gray_AeraremoveScratch_2;
					else
						removeScratchFlag = gray_AeraremoveScratch_1;
					if (removeScratch < removeScratchFlag && max((x_rt - x_lt - 2), (y_rt - y_lt - 2))>grayColor_LenthMax_LowLimit_1 && min((x_rt - x_lt - 2), (y_rt - y_lt - 2)) < grayColor_LenthMin_HighLimit_1)  //lackscratchth
					{
						if (sideLightStddev < 3) {
							result = true;
							CvPoint top_lef4 = cvPoint(x_lt, y_lt);
							CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
							rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
							string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
							putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, 8, 0);
							//imwrite("D:\\lackline-information_gray.bmp", img_gray);
							break;

						}

					}
					else if (removeScratch < removeScratchFlag && max((x_rt - x_lt - 2), (y_rt - y_lt - 2))>grayColor_LenthMax_LowLimit_2 && min((x_rt - x_lt - 2), (y_rt - y_lt - 2)) < grayColor_LenthMin_HighLimit_2)
					{
						if (sideLightStddev < 3) {
							result = true;
							CvPoint top_lef4 = cvPoint(x_lt, y_lt);
							CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
							rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
							string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
							putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, 8, 0);
							//imwrite("D:\\lackline-information_gray.bmp", img_gray);
							break;
						}
					}
				}
			}
		}
	}

	vector<Rect> boundRect_lackLine(contours_lackline.size());
	{
		for (vector<int>::size_type i = 0; i < contours_lackline.size(); i++)
		{

			double area = contourArea(contours_lackline[i]);

			RotatedRect rect = minAreaRect(contours_lackline[i]);  //包覆轮廓的最小斜矩形
			Point p = rect.center;
			//按图片坐标系为准，左上角原点,上侧Y轴，左侧X轴
			X_list.push_back(p.y);
			Y_list.push_back(p.x);

			if (area > 30 && area < 50000)
			{
				Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
				drawContours(temp_mask, contours_lackline, i, 255, FILLED, 8);
				boundRect_lackLine[i] = boundingRect(Mat(contours_lackline[i]));
				float w = boundRect_lackLine[i].width;
				float h = boundRect_lackLine[i].height;
				int X_1 = boundRect_lackLine[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect_lackLine[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect_lackLine[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect_lackLine[i].br().y;//矩形右下角Y坐标值
				Moments m = moments(contours_lackline[i]);
				int x_point = int(m.m10 / m.m00);
				int y_point = int(m.m01 / m.m00);
				if (x_point < 0 || x_point >= imageBinary.cols || y_point < 0 || y_point >= imageBinary.rows)
				{
					continue;
				}
				double longShortRatio = max(h / w, w / h);
				double 	longShortlowerth = 8;//11 4
				int shortHigherth = 60;
				//if (longShortRatio > longShortlowerth && min(w, h) < shortHigherth && max(w, h) > 40)//长宽比，长度，宽度限制
				if (longShortRatio > longShortlowerth && min(w, h) < shortHigherth && max(w, h) > 40)//长宽比，长度，宽度限制
				{
					int border = 3;//选定框边界宽度
					int x_lt = X_1 - border;
					//越界保护
					if (x_lt < 0)
					{
						x_lt = 0;
					}
					int y_lt = Y_1 - border;
					if (y_lt < 0)
					{
						y_lt = 0;
					}
					int x_rt = X_2 + border;
					if (x_rt > img_gray.size[1] - 1)
					{
						x_rt = img_gray.size[1] - 1;
					}
					int y_rt = Y_2 + border;
					if (y_rt > img_gray.size[0] - 1)
					{
						y_rt = img_gray.size[0] - 1;
					}
					//侧光图排除明显划痕
					Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 5, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
					Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
					double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
					double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
					double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                        //排除侧光图贴膜划痕的参数
					double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
					cv::Mat meanGray;
					cv::Mat stdDev;
					cv::meanStdDev(sidelightSuspect, meanGray, stdDev);
					double sideLightStddev = stdDev.at<double>(0, 0);
					double removeScratchFlag;
					if (removeScratchArea > 10000)
						removeScratchFlag = 1.3;
					else
						removeScratchFlag = 1.05;
					if (removeScratch < removeScratchFlag && max((x_rt - x_lt - 2), (y_rt - y_lt - 2))>95 && min((x_rt - x_lt - 2), (y_rt - y_lt - 2)) < 15)  //lackscratchth
					{
						if (sideLightStddev < 2.5) {
							result = true;
							CvPoint top_lef4 = cvPoint(x_lt, y_lt);
							CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
							rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
							string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
							putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, 8, 0);
							//imwrite("D:\\lackline-information_gray.bmp", img_gray);
							break;

						}

					}
					else if (removeScratch < removeScratchFlag && max((x_rt - x_lt - 2), (y_rt - y_lt - 2))>500 && min((x_rt - x_lt - 2), (y_rt - y_lt - 2)) < 30)
					{
						if (sideLightStddev < 2.5) {
							result = true;
							CvPoint top_lef4 = cvPoint(x_lt, y_lt);
							CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
							rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
							string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
							putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, 8, 0);
							//imwrite("D:\\lackline-information_gray.bmp", img_gray);
							break;
						}
					}
				}
			}
		}
	}
	stack<int> x_stack;
	stack<int> y_stack;
	if (result == false)
	{
		int min_x;
		int max_x;
		for (vector<int>::size_type i = 0; i < X_list.size(); i++)
		{
			int temp_x = X_list.at(i);
			int temp_y = Y_list.at(i);

			if (count_x.at(temp_x) > 15)//26
			{
				if (!(x_stack.empty()))
				{
					int temp = x_stack.top();
					if (temp != temp_x)
					{
						//x_stack.pop();
						x_stack.push(temp_x);
					}
				}
				else
					x_stack.push(temp_x);
			}
			else
			{
				count_x[temp_x]++;
			}
			if (count_y.at(temp_y) > 15)//26
			{
				if (!(y_stack.empty()))
				{
					int temp = y_stack.top();
					if (temp != temp_y)
					{
						//y_stack.pop();
						y_stack.push(temp_y);
					}
				}
				else
					y_stack.push(temp_y);
			}
			else
			{
				count_y[temp_y]++;
			}
		}
		Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
		while (!(x_stack.empty()) && result == false)
		{
			int index = x_stack.top();
			x_stack.pop();

			min_x = Y_list.at(0);
			max_x = Y_list.at(0);
			for (vector<int>::size_type i = 0; i < X_list.size(); i++)
			{
				int temp_y = Y_list.at(i);
				if (X_list.at(i) == index)
				{
					if (temp_y < min_x)
					{
						min_x = temp_y;
					}
					if (temp_y > max_x)
					{
						max_x = temp_y;
					}
				}
			}
			temp_mask(Rect(min_x, index, max_x - min_x, 1)) = uchar(255);
			int X_1 = min_x;//矩形左上角X坐标值
			int Y_1 = index;//矩形左上角Y坐标值
			int X_2 = max_x;//矩形右下角X坐标值
			int Y_2 = index;//矩形右下角Y坐标值
			int border = 3;//选定框边界宽度
			int x_lt = X_1 - border;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border;
			if (x_rt > img_gray.size[1] - 1)
			{
				x_rt = img_gray.size[1] - 1;
			}
			int y_rt = Y_2 + border;
			if (y_rt > img_gray.size[0] - 1)
			{
				y_rt = img_gray.size[0] - 1;
			}
			//侧光图排除明显划痕
			Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
			Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
			double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
			double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
			double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                   //排除侧光图贴膜划痕的参数
			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
			double removeScratchFlag;
			cv::Mat meanGray;
			cv::Mat stdDev;
			for (int k = 0; k < sidelightSuspect.cols; k++)
			{
				for (int t = 0; t < sidelightSuspect.rows; t++)
				{
					if (sidelightSuspect.at<uchar>(t, k) == 0)
					{
						sidelightSuspect.at<uchar>(t, k) = meanGrayout_Suspect;
					}
				}
			}
			cv::meanStdDev(sidelightSuspect, meanGray, stdDev);
			double sideLightStddev = stdDev.at<double>(0, 0);
			if (removeScratchArea > 10000)
				removeScratchFlag = 5;
			else
				removeScratchFlag = 3;
			if (removeScratch < removeScratchFlag&& sideLightStddev < 5)  //lackscratchth
			{
				result = true;
				CvPoint top_lef4 = cvPoint(x_lt, y_lt);
				CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
				rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
				//string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
				//putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, 8, 0);
				//imwrite("D:\\lackline.bmp", img_gray);
			}
		}

		while (!(y_stack.empty()) && result == false)
		{
			int index = y_stack.top();
			y_stack.pop();

			min_x = X_list.at(0);
			max_x = X_list.at(0);
			for (vector<int>::size_type i = 0; i < X_list.size(); i++)
			{
				int temp_x = X_list.at(i);
				if (Y_list.at(i) == index)
				{

					if (temp_x < min_x)
					{
						min_x = temp_x;
					}
					if (temp_x > max_x)
					{
						max_x = temp_x;
					}
				}
			}
			temp_mask(Rect(index, min_x, 1, max_x - min_x)) = uchar(255);
			int X_1 = index;//矩形左上角X坐标值
			int Y_1 = min_x;//矩形左上角Y坐标值
			int X_2 = index;//矩形右下角X坐标值
			int Y_2 = max_x;//矩形右下角Y坐标值
			int border = 3;//选定框边界宽度
			int x_lt = X_1 - border;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border;
			if (x_rt > img_gray.size[1] - 1)
			{
				x_rt = img_gray.size[1] - 1;
			}
			int y_rt = Y_2 + border;
			if (y_rt > img_gray.size[0] - 1)
			{
				y_rt = img_gray.size[0] - 1;
			}
			//侧光图排除明显划痕
			Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
			Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
			double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
			double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
			double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                   //排除侧光图贴膜划痕的参数
			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
			double removeScratchFlag;
			cv::Mat meanGray;
			cv::Mat stdDev;
			for (int k = 0; k < sidelightSuspect.cols; k++)
			{
				for (int t = 0; t < sidelightSuspect.rows; t++)
				{
					if (sidelightSuspect.at<uchar>(t, k) == 0)
					{
						sidelightSuspect.at<uchar>(t, k) = meanGrayout_Suspect;
					}
				}
			}
			cv::meanStdDev(sidelightSuspect, meanGray, stdDev);
			double sideLightStddev = stdDev.at<double>(0, 0);
			if (removeScratchArea > 10000)
				removeScratchFlag = 5;
			else
				removeScratchFlag = 3;
			if (removeScratch < removeScratchFlag&& sideLightStddev < 3)  //lackscratchth
			{
				result = true;
				CvPoint top_lef4 = cvPoint(x_lt, y_lt);
				CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
				rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
				//string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
				//putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, 8, 0);
				//imwrite("D:\\lackline.bmp", img_gray);
			}
		}

	}
	if (result == true)
	{
		fprintf(pOutFile, "%s,", "灰底彩色下");
		*mresult = img_gray;
		*causecolor = "少线";
		result = true;
	}
	return result;

}


bool sortContoursLength(vector<cv::Point> contour1, vector<cv::Point> contour2) {
	return (boundingRect(contour1).width > boundingRect(contour2).width);
}

double CalcMHWScore(vector<int> scores)
{
	size_t size = scores.size();

	if (size == 0)
	{
		return 0;  // Undefined, really.
	}
	else
	{
		sort(scores.begin(), scores.end());
		if (size % 2 == 0)
		{
			return (scores[size / 2 - 1] + scores[size / 2]) / 2;
		}
		else
		{
			return scores[size / 2];
		}
	}
}
bool lack_line_rgb_My(Mat src_red, Mat ceguang, Mat src_red_old,Mat Shield_MY_IN,Mat* mresult, string* causecolor, string name) {
	bool result = false;
	Mat image_rgb = src_red.clone();
	Mat image_red = src_red.clone();
	Mat kernel1 = getGaborKernel(Size(3, 3), 2 * CV_PI, CV_PI / 2, 1.0, 1.0, 0, CV_32F);//求卷积核
	float sum = 0.0;
	for (int i = 0; i < kernel1.rows; i++)
	{
		for (int j = 0; j < kernel1.cols; j++)
		{
			sum = sum + kernel1.ptr<float>(i)[j];
		}
	}
	Mat mmm = kernel1 / sum;
	Mat kernel2 = getGaborKernel(Size(3, 3), 2 * CV_PI, 0, 1.0, 1.0, 0, CV_32F);
	float sum2 = 0.0;
	for (int i = 0; i < kernel2.rows; i++)
	{
		for (int j = 0; j < kernel2.cols; j++)
		{
			sum2 = sum2 + kernel2.ptr<float>(i)[j];
		}
	}
	Mat mmm2 = kernel2 / sum2;
	Mat img_4, img_5;
	filter2D(image_red, img_4, -1, mmm);//卷积运算
	filter2D(img_4, image_red, -1, mmm2);
	filter2D(src_red, img_4, -1, mmm);//卷积运算
	filter2D(img_4, src_red, -1, mmm2);
	Mat gray_ceguang = ceguang.clone();
	//medianBlur(gray_ceguang, gray_ceguang, 9);
	medianBlur(ceguang, ceguang, 9);
	Mat imageBinaryblue;//分割二值图
	Mat imageBinarygreen;//分割二值图
	Mat imageBinaryred;//分割二值图
	Mat imageBinaryblue2;//分割二值图
	Mat imageBinarygreen2;//分割二值图
	Mat imageBinaryred2;//分割二值图
	Mat ceimageBinary;//分割二值图
	int dividlinebg = 0;
	int dividlinegr = 0;
	//QTime startTime1 = QTime::currentTime();

	Mat imageBinary = Mat::zeros(src_red.size(), CV_8UC1);
	Mat imageBinary2 = Mat::zeros(src_red.size(), CV_8UC1);
	int ratio = 3;
	int kernel_size = 3;
	int lowThreshold = 4;
	//RGB区域划分
	Canny(image_red, imageBinary2, lowThreshold, lowThreshold * ratio, kernel_size);
	imageBinary2(Rect(0, 0, imageBinary2.cols, 100)) = 0;
	imageBinary2(Rect(0, imageBinary2.rows - 100, imageBinary2.cols, 100)) = 0;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(imageBinary2, imageBinary2, MORPH_CLOSE, element);   //开运算形态学操作。可以减少噪点
	Vec4f leftLine_Fit, rightLine_Fit;				    //左右拟合直线数据
	bitwise_and(image_red, ~imageBinary2, imageBinary);
	vector<vector<Point>> contours_lacklinebg, contours_lacklinegr;
	findContours(imageBinary2(Rect(900, 0, 200, imageBinary2.rows)), contours_lacklinebg, CV_RETR_LIST, CHAIN_APPROX_NONE);   //外围轮廓2022.6.13
	sort(contours_lacklinebg.begin(), contours_lacklinebg.end(), compareContourSizes);
	if (contours_lacklinebg.size() != 0) {
		fitLine(contours_lacklinebg[0], leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
		dividlinebg = 900 + leftLine_Fit[2];
	}
	else {
		dividlinebg = 1000;
	}
	findContours(imageBinary2(Rect(1900, 0, 200, imageBinary2.rows)), contours_lacklinegr, CV_RETR_LIST, CHAIN_APPROX_NONE);   //外围轮廓2022.6.13
	sort(contours_lacklinegr.begin(), contours_lacklinegr.end(), compareContourSizes);
	if (contours_lacklinegr.size() != 0) {
		fitLine(contours_lacklinegr[0], rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //右侧拟合直线
		dividlinegr = 1900 + rightLine_Fit[2];
	}
	else {
		dividlinegr = 2000;
	}
	static int n = 0;

	//imwrite("D:\\SXbgr\\imageBinary_canny"+to_string(n+1)+".bmp",imageBinary2);
	cout << "dividlinegr" << dividlinegr << endl;
	cout << "dividlinebg" << dividlinebg << endl;

	Mat blue = src_red(Rect(0, 0, dividlinebg, src_red.rows)).clone();
	Mat green = src_red(Rect(dividlinebg, 0, dividlinegr - dividlinebg, src_red.rows)).clone();
	Mat red = src_red(Rect(dividlinegr, 0, src_red.cols - dividlinegr, src_red.rows)).clone();

	//Mat red_N = src_red_old(Rect(dividlinegr, 0, src_red.cols - dividlinegr, src_red.rows)).clone();

	//Ptr<CLAHE> clahe_R = createCLAHE(1, Size(1, 1));

	//clahe_R->apply(red_N, red_N);   //整图增强

	//red_N = Gabor5(red_N);
	Mat shieldMask;
	Mat shieldMask2;
	Mat img_blue, img_green, img_red;
	Mat element1 = getStructuringElement(MORPH_RECT, Size(7, 7));
	threshold(red, shieldMask, 125, 255, CV_THRESH_BINARY);//120   130
	dilate(shieldMask, shieldMask, element1);

	bitwise_and(red, ~shieldMask, red);

	//bitwise_and(red_N, ~shieldMask, red_N);

	medianBlur(blue, img_blue, 3);

	medianBlur(green, img_green, 3);

	medianBlur(red, img_red, 3);

	adaptiveThresholdCustom(img_blue, imageBinaryblue, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 33, 2, 1, true);//35   31
	adaptiveThresholdCustom(img_green, imageBinarygreen, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 15, 2, 1, true);//17    15
	//adaptiveThreshold(img_green, imageBinarygreen, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 3);//lackline_bolckSize, lackline_delta
	//adaptiveThresholdCustom(img_red, imageBinaryred, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 33, 2, 1, true);//35   33
	adaptiveThresholdCustom(img_red, imageBinaryred, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 33, 2, 1, true);//35   33
	Ptr<CLAHE> clahe = createCLAHE(0.5, Size(20, 20));

	clahe->apply(img_blue, img_blue);   //整图增强
	

	adaptiveThresholdCustom(img_blue, imageBinaryblue2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 33, -2, 1, true);  //31
	adaptiveThresholdCustom(img_green, imageBinarygreen2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, -2, 1, true);//15
	adaptiveThresholdCustom(img_red, imageBinaryred2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 33, -2, 1, true);

	medianBlur(imageBinarygreen2, imageBinarygreen2, 5);

	bitwise_or(imageBinaryblue, imageBinaryblue2, imageBinaryblue);
	bitwise_or(imageBinarygreen, imageBinarygreen2, imageBinarygreen);
	bitwise_or(imageBinaryred, imageBinaryred2, imageBinaryred);

	imageBinaryblue(Rect(0, 0, 40, imageBinaryblue.rows)) = uchar(0);
	imageBinaryblue(Rect(imageBinaryblue.cols - 40, 0, 40, imageBinaryblue.rows)) = uchar(0);
	imageBinarygreen(Rect(0, 0, 40, imageBinarygreen.rows)) = uchar(0);
	imageBinarygreen(Rect(imageBinarygreen.cols - 40, 0, 40, imageBinarygreen.rows)) = uchar(0);
	imageBinaryred(Rect(0, 0, 40, imageBinaryred.rows)) = uchar(0);
	imageBinaryred(Rect(imageBinaryred.cols - 40, 0, 40, imageBinaryred.rows)) = uchar(0);
	imageBinaryblue.copyTo(imageBinary(Rect(0, 0, dividlinebg, src_red.rows)));
	imageBinarygreen.copyTo(imageBinary(Rect(dividlinebg, 0, dividlinegr - dividlinebg, src_red.rows)));
	imageBinaryred.copyTo(imageBinary(Rect(dividlinegr, 0, src_red.cols - dividlinegr, src_red.rows)));
	imageBinary(Rect(0, 0, imageBinary.cols, 40)) = uchar(0);
	imageBinary(Rect(0, imageBinary.rows - 40, imageBinary.cols, 40)) = uchar(0);
	//erode(imageBinary, imageBinary, element);
	morphologyEx(imageBinary, imageBinary, MORPH_OPEN, element);   //开运算形态学操作。可以减少噪点
	imageBinary(Rect(imageBinary.cols - 250, 0, 250, 250)) = uchar(0);//去除易撕贴部分域
	imageBinary(Rect(imageBinary.cols - 180, 500, 180, 600)) = uchar(0);

	vector<vector<Point>> contours_lacklinefit;

	vector<int> X_list;
	vector<int> Y_list;
	vector<int> count_x;
	vector<int> count_y;
	count_x.resize(imageBinary.rows);
	count_y.resize(imageBinary.cols);
	//侧光排除
	Mat element21 = getStructuringElement(MORPH_RECT, Size(55, 55));//闭操作结构元素

	Mat ceimageBinary2;
	adaptiveThresholdCustom(ceguang, ceimageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 2, 1, true);//21 5  LQY:17 3


	vector<vector<Point>> contours_lacklinece;

	findContours(ceimageBinary, contours_lacklinece, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);   //外围轮廓2022.2.8
	vector<Rect> boundRect_lackLinece(contours_lacklinece.size());
	Mat temp_maskce = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);

	for (vector<int>::size_type i = 0; i < contours_lacklinece.size(); i++) {
		double area = contourArea(contours_lacklinece[i]);

		RotatedRect rect = minAreaRect(contours_lacklinece[i]);  //包覆轮廓的最小斜矩形
		if (area > 400) {
			Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
			drawContours(temp_mask, contours_lacklinece, i, 255, FILLED, 8);
			boundRect_lackLinece[i] = boundingRect(Mat(contours_lacklinece[i]));
			float w = boundRect_lackLinece[i].width;
			float h = boundRect_lackLinece[i].height;
			int X_1 = boundRect_lackLinece[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect_lackLinece[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect_lackLinece[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect_lackLinece[i].br().y;//矩形右下角Y坐标值
			double longShortRatio = max(h / w, w / h);
			double 	longShortlowerth = 10;//11 4
			int shortHigherth = 100;

			if (longShortRatio > 3 && min(w, h) < 250 && max(w, h) > 50)//长宽比，长度，宽度限制
			{
				drawContours(temp_maskce, contours_lacklinece, i, 255, FILLED, 8);
			}
		}

	}
	Mat temp_maskce2 = temp_maskce(Rect(100, 100, temp_maskce.cols - 200, temp_maskce.rows - 200));
	dilate(temp_maskce2, temp_maskce2, element21);
	temp_maskce2.copyTo(ceimageBinary(Rect(100, 100, temp_maskce.cols - 200, temp_maskce.rows - 200)));
	bitwise_and(~ceimageBinary, imageBinary, imageBinary);
	

	Mat element2 = getStructuringElement(MORPH_RECT, Size(7, 7));
	vector<vector<Point>> contours_lackline;

	//屏蔽区处理
	bitwise_and(imageBinary, Shield_MY_IN, imageBinary);

	findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);

	sort(contours_lackline.begin(), contours_lackline.end(), compareContourAreas);
	vector<Rect> boundRect_lackLine(contours_lackline.size());
	for (vector<int>::size_type i = 0; i < contours_lackline.size(); i++)
	{
		RotatedRect rect = minAreaRect(contours_lackline[i]);  //包覆轮廓的最小斜矩形
		Point p = rect.center;
		double angle = rect.angle;
		double area = contourArea(contours_lackline[i]);
		double width1 = min(rect.size.height, rect.size.width);
		X_list.push_back(p.y);
		Y_list.push_back(p.x);

		// 计算角度偏差
		double angleBias = abs(abs(angle / 90) - 1);
		double verticalThresh = 0.1;

		
		bool notVertical = (angleBias < 1 - verticalThresh) && (angleBias > verticalThresh);
		//宽度35改为15
		if (width1 < 35 && area > 100 && area < 500000 && (angle < -80 || angle > -10))//-80  -10
		{
			//按图片坐标系为准，左上角原点,上侧Y轴，左侧X轴

			Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
			drawContours(temp_mask, contours_lackline, i, 255, FILLED, 8);
			boundRect_lackLine[i] = boundingRect(Mat(contours_lackline[i]));
			float w = boundRect_lackLine[i].width;
			float h = boundRect_lackLine[i].height;
			int X_1 = boundRect_lackLine[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect_lackLine[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect_lackLine[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect_lackLine[i].br().y;//矩形右下角Y坐标值
			if (w < h) {
				continue;//水平方向（|）少线去除
			}
			Moments m = moments(contours_lackline[i]);
			int x_point = int(m.m10 / m.m00);
			int y_point = int(m.m01 / m.m00);
			if (x_point < 0 || x_point >= imageBinary.cols || y_point < 0 || y_point >= imageBinary.rows)
			{
				continue;
			}
			double longShortRatio = max(h / w, w / h);
			if (longShortRatio >= 2 && min(w, h) < 25 && max(w, h) > 60)//长宽比，长度，宽度限制
			{
				int border = 3;//选定框边界宽度
				int x_lt = X_1 - border;
				//越界保护
				if (x_lt < 0)
				{
					x_lt = 0;
				}
				int y_lt = Y_1 - border;
				if (y_lt < 0)
				{
					y_lt = 0;
				}
				int x_rt = X_2 + border;
				if (x_rt > src_red.size[1] - 1)
				{
					x_rt = src_red.size[1] - 1;
				}
				int y_rt = Y_2 + border;
				if (y_rt > src_red.size[0] - 1)
				{
					y_rt = src_red.size[0] - 1;
				}
				//侧光图排除明显划痕
				//Mat sidelightSuspect1 = gray_ceguang(boundRect_lackLine[i]);
				Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
				Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
				//Mat mask1 = temp_mask(boundRect_lackLine[i]);             //侧光图像疑似贴膜划痕掩膜


				double meanOut;//缺陷外围灰度均值

				double meanIn;//缺陷区域灰度均值
				Mat maskGray;//做掩膜后的图像

				int grayValueSum = 0;//像素灰度总和
				double graySqrtSum = 0.0;//标准差分母
				double DefStrDev;//侧光图标准差1
				int pixelsNum = 0;//像素点个数
				bitwise_and(sidelightSuspect, mask, maskGray);//侧光图做掩膜
				//计算侧光图上缺陷区域灰度均值
				int Rangece;
				int MaxGrayce = 0;
				int MinGrayce = 255;
				for (int i = 0; i < maskGray.cols; i++)
				{
					for (int j = 0; j < maskGray.rows; j++)
					{
						if (maskGray.at<uchar>(j, i) > 0)
						{
							if (maskGray.at<uchar>(j, i) > MaxGrayce)
							{
								MaxGrayce = maskGray.at<uchar>(j, i);
							}
							if (maskGray.at<uchar>(j, i) < MinGrayce)
							{
								MinGrayce = maskGray.at<uchar>(j, i);
							}
							grayValueSum += maskGray.at<uchar>(j, i);
							pixelsNum++;
						}
					}
				}
				meanIn = grayValueSum / (float)pixelsNum;
				Rangece = MaxGrayce - MinGrayce;
				//计算标准差
				for (int i = 0; i < maskGray.cols; i++)
				{
					for (int j = 0; j < maskGray.rows; j++)
					{
						if (maskGray.at<uchar>(j, i) > 0)
						{
							graySqrtSum += (maskGray.at<uchar>(j, i) - meanIn)*(maskGray.at<uchar>(j, i) - meanIn);
						}
					}
				}
				DefStrDev = sqrt(graySqrtSum / pixelsNum);


				//计算侧光图上缺陷外围灰度均值
				grayValueSum = 0;
				pixelsNum = 0;
				bitwise_and(sidelightSuspect, ~mask, maskGray);
				for (int i = 0; i < maskGray.cols; i++)
				{
					for (int j = 0; j < maskGray.rows; j++)
					{
						if (maskGray.at<uchar>(j, i) <= meanIn && maskGray.at<uchar>(j, i) > 0)
						{
							grayValueSum += maskGray.at<uchar>(j, i);
							pixelsNum++;
						}
					}
				}
				meanOut = grayValueSum / (float)pixelsNum;
				double removeScratch = meanIn - meanOut;//缺陷区域与周围灰度差值(整体性)

				Mat white_defect = image_rgb(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//白底疑似缺陷区域图像
				double meanwhite_in = mean(white_defect, mask)[0];
				double meanwhite_out = mean(white_defect, ~mask)[0];
				double meanwhite_diff = meanwhite_in - meanwhite_out;

				//极差计算
				Mat maskGrayRange;//底色图内部
				bitwise_and(white_defect, mask, maskGrayRange);//彩底图做掩膜
				int Range;
				int MaxGray = 0;
				int MinGray = 255;
				//计算彩底图上缺陷区域极差
				for (int i = 0; i < maskGrayRange.cols; i++)
				{
					for (int j = 0; j < maskGrayRange.rows; j++)
					{
						if (maskGrayRange.at<uchar>(j, i) > 0)
						{
							if (maskGrayRange.at<uchar>(j, i) > MaxGray)
							{
								MaxGray = maskGrayRange.at<uchar>(j, i);
							}
							if (maskGrayRange.at<uchar>(j, i) < MinGray)
							{
								MinGray = maskGrayRange.at<uchar>(j, i);
							}
						}
					}
				}
				Range = MaxGray - MinGray;
				//彩底图内标准差
				double graySqrtSumrgb = 0.0;//标准差分母
				double sideLightStddevrgb;//彩底图标准差1
				int pixelsNumbgr = 0;//像素点个数
				for (int i = 0; i < maskGrayRange.cols; i++)
				{
					for (int j = 0; j < maskGrayRange.rows; j++)
					{
						if (maskGrayRange.at<uchar>(j, i) > 0)
						{
							graySqrtSumrgb = (maskGrayRange.at<uchar>(j, i) - meanwhite_in)*(maskGrayRange.at<uchar>(j, i) - meanwhite_in);
							pixelsNumbgr++;
						}
					}
				}
				sideLightStddevrgb = sqrt(graySqrtSumrgb / pixelsNumbgr);

				//计算
				cv::Mat meanGray;
				cv::Mat stdDev;
				cv::Mat meanGray1;
				cv::Mat stdDev1;
				cv::meanStdDev(maskGrayRange, meanGray, stdDev);
				cv::meanStdDev(sidelightSuspect, meanGray1, stdDev1);
				double meanInrgb2 = meanGray.at<double>(0, 0);//彩底图均值2
				double sideLightStddevrgb2 = stdDev.at<double>(0, 0);//彩色图标准差2
				double meanIn1 = meanGray1.at<double>(0, 0);//侧光图均值2
				double sideLightStddev1 = stdDev1.at<double>(0, 0);//侧光图标准差2
				double coeffGray = sideLightStddevrgb / meanwhite_in;//彩底图变异系数

				if (removeScratch <5 && abs(meanwhite_diff) >1.5)//lackscratchth 5 //sideLightStddev >3.2
				{
					if (meanwhite_diff > -3.5 && meanwhite_diff < -1.5 && sideLightStddevrgb >0 && sideLightStddevrgb < 0.072)//-3.2   -1.5   0  0.072
					{

					}
					else if (meanwhite_diff > -4 && meanwhite_diff < 1) {

					}
					else
					{
						//补丁1 距离两边在一定范围内
						
						if (X_1<200  ||  X_2>2750  )
						{
							cout << "white_celv1 " << "removeScratch:" << removeScratch << "meanwhite_diff:" << meanwhite_diff << endl;

							result = true;
							CvPoint top_lef4 = cvPoint(x_lt, y_lt);
							CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
							rectangle(image_rgb, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
							string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
							putText(image_rgb, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
							*mresult = image_rgb;
							//*mresult = src_red;
							*causecolor = "少线";
							break;
						}//补丁2(或关系) 中间白底 起始位于图像分割周围且具有一定宽度
						else if (((abs(dividlinebg - X_1) < 100) || (abs(dividlinegr - X_2) < 100))&& w > 100)
						{
							cout << "white_celv1 " << "removeScratch:" << removeScratch << "meanwhite_diff:" << meanwhite_diff << endl;

							result = true;
							CvPoint top_lef4 = cvPoint(x_lt, y_lt);
							CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
							rectangle(image_rgb, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
							string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
							putText(image_rgb, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
							*mresult = image_rgb;
							//*mresult = src_red;
							*causecolor = "少线";
							break;
						}
						else {
							continue;
						}
						
					}
				}
			}
		}
	}
	stack<int> x_stack;
	stack<int> y_stack;
	//短线补丁
	if (result == false)
	{
		int min_x;
		int max_x;
		for (vector<int>::size_type i = 0; i < X_list.size(); i++)
		{
			int temp_x = X_list.at(i);
			int temp_y = Y_list.at(i);

			// 相同x（图像的y坐标）坐标数量大于10的就算是合理的轮廓
			if (count_x.at(temp_x) >= 10)//26 >10
			{
				if (!(x_stack.empty()))
				{
					int temp = x_stack.top();
					if (temp != temp_x)
					{
						//x_stack.pop();
						x_stack.push(temp_x);
					}
				}
				else
					x_stack.push(temp_x);
			}
			else
			{
				count_x[temp_x]++;
			}
			if (count_y.at(temp_y) >= 16)//26
			{
				if (!(y_stack.empty()))
				{
					int temp = y_stack.top();
					if (temp != temp_y)
					{
						//y_stack.pop();
						y_stack.push(temp_y);
					}
				}
				else
					y_stack.push(temp_y);
			}
			else
			{
				count_y[temp_y]++;
			}
		}
		Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
		while (!(x_stack.empty()) && result == false)
		{
			int index = x_stack.top();
			x_stack.pop();//横坐标值

			min_x = Y_list.at(0);
			max_x = Y_list.at(0);
			//补丁 剔除离群异常坐标点
			vector<int> Y_Find_Inline;
			vector<int> Y_Find_Inline_End;
			vector<int> Y_Find_Inline_Diff;

			for (vector<int>::size_type i = 0; i < X_list.size(); i++)
			{
				if (X_list.at(i) == index)
				{
					Y_Find_Inline.push_back(Y_list.at(i));
				}
			}
			sort(Y_Find_Inline.begin(), Y_Find_Inline.end());
			double Test_cull = CalcMHWScore(Y_Find_Inline);

			for (int i = 0; i < Y_Find_Inline.size(); i++)
			{
				Y_Find_Inline_Diff.push_back(abs(int(Test_cull) - Y_Find_Inline.at(i)));
			}

			double Test_cull_M = CalcMHWScore(Y_Find_Inline_Diff);

			for (int i = 0; i < Y_Find_Inline.size(); i++)
			{
				if(Y_Find_Inline_Diff.at(i) < 4* Test_cull_M)
				{
					Y_Find_Inline_End.push_back(Y_Find_Inline.at(i));
				}
				
			}
			max_x = *max_element(Y_Find_Inline_End.begin(), Y_Find_Inline_End.end());
			min_x = *min_element(Y_Find_Inline_End.begin(), Y_Find_Inline_End.end());
			//for (vector<int>::size_type i = 0; i < X_list.size(); i++)
			//{
			//	int temp_y = Y_list.at(i);
			//	// 获取相同x坐标轮廓的最大y坐标作为矩形框一边
			//	if (X_list.at(i) == index)
			//	{
			//		if (temp_y < min_x)
			//		{
			//			min_x = temp_y;
			//		}
			//		if (temp_y > max_x)
			//		{
			//			max_x = temp_y;
			//		}
			//	}
			//}
			temp_mask(Rect(min_x, index, max_x - min_x, 1)) = uchar(255);
			int X_1 = min_x;//矩形左上角X坐标值
			int Y_1 = index;//矩形左上角Y坐标值
			int X_2 = max_x;//矩形右下角X坐标值
			int Y_2 = index;//矩形右下角Y坐标值
			int border = 3;//选定框边界宽度
			int x_lt = X_1 - border;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border;
			if (x_rt > src_red.size[1] - 1)
			{
				x_rt = src_red.size[1] - 1;
			}
			int y_rt = Y_2 + border;
			if (y_rt > src_red.size[0] - 1)
			{
				y_rt = src_red.size[0] - 1;
			}


			//两边区域灰度差计算
			//cout<<"=============grayJianSuan=============="<<endl;
			double GrayDiffSum = 0;
			int totalGrayDiffNum = 0;
			bool flag = false;
			
			
			//左侧
			int leftIndex = 20;
			int lenght = min_x + (max_x - min_x) / 4 - 5;
			if (lenght > 1000)
			{
				lenght = 900;
			}
			for (int i = 0; leftIndex <= lenght; i++)
			{
				leftIndex += i * 5;
				Mat srcArea = image_rgb(Rect(leftIndex, y_lt + 1, 5, y_rt - y_lt - 2));
				Mat srcArea2;
				if (y_lt - y_rt + y_lt + 3 > 0)
				{
					srcArea2 = image_rgb(Rect(leftIndex, y_lt - y_rt + y_lt + 3, 5, y_rt - y_lt - 2));//上侧
				}
				else {
					srcArea2 = image_rgb(Rect(leftIndex, y_rt - 1, 5, y_rt - y_lt - 2));//下侧
				}
				double grayMean = mean(srcArea)[0];
				double grayMean2 = mean(srcArea2)[0];
				double grayDiff = grayMean - grayMean2;
				//cout<<"grayDiff_left"<<grayDiff<<endl;
				//QString j = QString::number(grayDiff);
				//file.write("grayDiff_left:");
				//file.write(j.toStdString().c_str());
				//file.write("    ");
				totalGrayDiffNum++;
				//if (abs(grayDiff) > 0.5 )   //参数待定
				//{
				GrayDiffSum = GrayDiffSum + abs(grayDiff);
				//}
			}
			//cout<<"GrayDiffSum:"<<GrayDiffSum<<endl;
			//cout<<"totalGrayDiffNum:"<<totalGrayDiffNum<<endl;
			double MeanGray = double(GrayDiffSum) / totalGrayDiffNum;
			//cout<<"MeanGray:"<<MeanGray<<endl;
			if (MeanGray > 0.75)
			{
				flag = true;
			}
			//file.write("\n");
			GrayDiffSum = 0.0;
			totalGrayDiffNum = 0;
			MeanGray = 0.0;
			if (flag == false)
			{
				//右侧
				int rightIndex = image_red.cols - 25 - 5; //刘海屏要区分

				int ScreenMode = 3; //刘海屏

				if (ScreenMode == 3 && y_lt + 1 >= 500 && y_lt + 1 <= 900)
				{
					rightIndex = image_red.cols - 130 - 5;
				}
				int lenght = min_x + x_rt - x_lt - 4 - (x_rt - x_lt - 4) / 4;
				if (lenght < 2000)
				{
					lenght = 2100;
				}

				for (int i = 0; rightIndex >= lenght + 5; i++)
				{
					rightIndex = rightIndex - i * 5;
					Mat srcArea = image_rgb(Rect(rightIndex, y_lt + 1, 5, y_rt - y_lt - 2));
					Mat srcArea2;
					if (y_lt - y_rt + y_lt + 3 > 0)
					{
						srcArea2 = image_rgb(Rect(rightIndex, y_lt - y_rt + y_lt + 3, 5, y_rt - y_lt - 2));
					}
					else {
						srcArea2 = image_rgb(Rect(rightIndex, y_rt - 1, 5, y_rt - y_lt - 2));
					}
					double grayMean = mean(srcArea)[0];
					double grayMean2 = mean(srcArea2)[0];
					double grayDiff = grayMean - grayMean2;
					//cout<<"grayDiff_right"<<grayDiff<<endl;
					//QString j = QString::number(grayDiff);
					
					totalGrayDiffNum++;
					//if (abs(grayDiff) > 0.5 )   //参数待定
					//{
					GrayDiffSum = GrayDiffSum + abs(grayDiff);
					//}
				}
				//cout<<"GrayDiffSum:"<<GrayDiffSum<<endl;
				//cout<<"totalGrayDiffNum:"<<totalGrayDiffNum<<endl;
				MeanGray = double(GrayDiffSum) / totalGrayDiffNum;
				//cout<<"MeanGray2:"<<MeanGray<<endl;
				if (MeanGray > 0.75)
				{
					flag = true;
				}
			}
			totalGrayDiffNum = 0;
			

			if (flag == true)
			{
				//侧光图排除明显划痕
				Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
				Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜

				double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
				double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
				double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                   //排除侧光图贴膜划痕的参数
				double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
				double removeScratchFlag;
				if (removeScratchArea > 10000)
					removeScratchFlag = 5;
				else
					removeScratchFlag = 3;
				cout << "000000000000:" << removeScratch << endl;
				if (abs(removeScratch) < removeScratchFlag)  //lackscratchth
				{
					if (x_lt < 200 || x_rt>2750)
					{
						result = true;
						CvPoint top_lef4 = cvPoint(x_lt, y_lt);
						CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
						rectangle(image_rgb, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
					}
					else if (((abs(dividlinebg - x_lt) < 100) || (abs(dividlinegr - x_rt) < 100)) && (x_rt- x_lt) > 100)
					{
						result = true;
						CvPoint top_lef4 = cvPoint(x_lt, y_lt);
						CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
						rectangle(image_rgb, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
					}
				}
			}
		}


		if (result == true)
		{
			*mresult = image_rgb;
			*causecolor = "少线";
			result = true;
		}
	}
	//边缘补丁
	if (result == false) {
		//只遍历右边缘轮廓 要求垂直 相对较短 数量较多
		//此处需要边界最大值 保证稳定性 后续引入
		vector<int>  X_Edge_list;
		vector<int>  X_End_Edge_list;
		vector<int>  Index_Edge_list;
		vector<int>  Index_Edge_End_list;

		int Find_Num = 0;//边缘数量
		for (vector<int>::size_type i = 0; i < contours_lackline.size(); i++)
		{
			Rect rect_Edge = boundingRect(contours_lackline[i]);  //包覆轮廓的最小斜矩形
			int X0_Edge = rect_Edge.x;
			int Y0_Edge = rect_Edge.y;
			int W0_Edge = rect_Edge.width;
			int H0_Edge = rect_Edge.height;
			if (abs(X0_Edge - 2900) < 100 && W0_Edge > 10 && W0_Edge> H0_Edge) //只要右边缘处少线
			{
				X_End_Edge_list.push_back(X0_Edge + W0_Edge);
				Index_Edge_list.push_back(i);
			}
		}
		if (!X_End_Edge_list.empty())
		{
			int Temp_Need_Max = *max_element(X_End_Edge_list.begin(), X_End_Edge_list.end());
			
			for (int i = 0; i < X_End_Edge_list.size(); i++)
			{
				if (X_End_Edge_list[i] == Temp_Need_Max)
				{
					Find_Num++;
					Index_Edge_End_list.push_back(Index_Edge_list[i]); //得到最终缺陷的索引号；
				}
			}
			if (Find_Num >= 2)
			{
				result = true;
				for (int T_temp = 0; T_temp < Find_Num; T_temp++)
				{
					Rect rect_Edge_Temp = boundingRect(contours_lackline[Index_Edge_End_list[T_temp]]);  //包覆轮廓的正交矩形

					/*int X0_Edge_Temp = rect_Edge_Temp.x;
					int Y0_Edge_Temp = rect_Edge_Temp.y;
					int W0_Edge_Temp = rect_Edge_Temp.width;
					int H0_Edge_Temp = rect_Edge_Temp.height;
					CvPoint top_lef4 = cvPoint(X0_Edge_Temp, Y0_Edge_Temp);
					CvPoint bottom_right4 = cvPoint(X0_Edge_Temp+W0_Edge_Temp, Y0_Edge_Temp+ H0_Edge_Temp);*/
					rectangle(image_rgb, rect_Edge_Temp,Scalar(255, 255, 255), 5, 8, 0);

				}
			}

			if (result == true)
			{
				*mresult = image_rgb;
				*causecolor = "少线";
				result = true;
			}
		}



	}
	return result;
}

Mat f_remove_Rcorner_bangs(Mat src)
{
	//边缘四侧处理
	src(Rect(0, 0, src.cols - 1, 20)) = uchar(0);               //去除四个边界的影响上边 10
	src(Rect(0, src.rows - 15, src.cols - 1, 15)) = uchar(0);    //下边
	src(Rect(0, 0, 50, src.rows - 1)) = uchar(0);              //左边
	src(Rect(src.cols - 15, 0, 15, src.rows - 1)) = uchar(0);  //右边

	//针对3/4屏幕的处理
	src(Rect(src.cols / 9, 0, 1, src.rows / 4)) = uchar(0);               //去除四个边界的影响上边 10
	src(Rect(0, src.rows / 4, src.cols / 9, 1)) = uchar(0);    //下边    7.31 1.77寸3->8
	src(Rect(src.cols / 9, src.rows * 3 / 4, 1, src.rows / 4)) = uchar(0);               //去除四个边界的影响上边 10
	src(Rect(0, src.rows * 3 / 4, src.cols / 9, 1)) = uchar(0);    //下边    7.31 1.77寸3->8
	src(Rect(src.cols * 8 / 9, 0, 1, src.rows / 3)) = uchar(0);               //去除四个边界的影响上边 10
	src(Rect(src.cols * 8 / 9, src.rows / 4, src.cols / 9, 1)) = uchar(0);    //下边    7.31 1.77寸3->8
	src(Rect(src.cols * 8 / 9, src.rows * 3 / 4, 1, src.rows / 4)) = uchar(0);               //去除四个边界的影响上边 10
	src(Rect(src.cols * 8 / 9, src.rows * 2 / 3, src.cols / 9, 1)) = uchar(0);    //下边


	Mat resultMat = src.clone();
	vector<vector<Point>> contours;
	findContours(src, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	vector<Rect> boundRect(contours.size());
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);

		boundRect[i] = boundingRect(Mat(contours[i]));
		if (area >= 900 &&
			((boundRect[i].x + boundRect[i].width < src.cols / 8 && boundRect[i].y + boundRect[i].height < src.rows / 3)//左上角
				|| (boundRect[i].x + boundRect[i].width < src.cols / 8 && boundRect[i].y > src.rows * 2 / 3)//左下角
				|| (boundRect[i].x > src.cols * 7 / 8 && boundRect[i].y + boundRect[i].height < src.rows / 3)//右上角
				|| (boundRect[i].x > src.cols * 7 / 8 && boundRect[i].y > src.rows * 2 / 3)//右下角
				|| (boundRect[i].x > src.cols * 7 / 8 && boundRect[i].height > src.rows / 8)))//刘海
		{
			drawContours(resultMat, contours, i, Scalar(0), CV_FILLED, 8, vector<Vec4i>(), 0, Point());

		}
	}
	return resultMat;
}
Mat f_remove_mark(Mat src, Mat ceguang)
{
	uint8_t F_strategy = 1; //策略参数
	Mat img_gray = src.clone();

	Mat resultMat = Mat::zeros(src.rows, src.cols, CV_8UC1);                                  //生成空白掩膜图像

	Mat imageBinary = Mat::zeros(src.rows, src.cols, CV_8UC1);//分割二值图
	// = Mat::zeros(src.rows, src.cols, CV_8UC1)
	//掩膜处理去除刘海干扰
	Mat th1 = Mat::zeros(src.rows, src.cols, CV_8UC1);
	threshold(img_gray, th1, 20, 255, CV_THRESH_BINARY);

	Mat structure_element2 = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(th1, th1, structure_element2); //膨胀边界

	bitwise_and(img_gray, th1, img_gray);


	adaptiveThreshold(img_gray, imageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 3);//lackline_bolckSize, lackline_delta

	/*vector<int> X_list;
	vector<int> Y_list;*/


	imageBinary(Rect(imageBinary.cols - 250, 0, 250, 250)) = uchar(0);//去除易撕贴部分域
	imageBinary(Rect(0, imageBinary.rows - 20, imageBinary.cols, 20)) = uchar(0);      //下边
	imageBinary(Rect(0, 0, 40, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	imageBinary(Rect(imageBinary.cols - 40, 0, 40, imageBinary.rows - 1)) = uchar(0);      //右边

	imageBinary = f_remove_Rcorner_bangs(imageBinary);

	vector<vector<Point>> contours_lackline;

	Mat structure_element_filter = (cv::Mat_<float>(11, 11) <<
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);//构建水平膨胀核 针对水平标注线

	structure_element_filter.convertTo(structure_element_filter, CV_8UC1);
	dilate(imageBinary, imageBinary, structure_element_filter);

	erode(imageBinary, imageBinary, structure_element_filter);    //将模板腐蚀一下,为了去除边界影响,否则相与过后会有白边

	findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE); //获得当前图像轮廓


	Mat Marke_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);                                  //生成空白掩膜图像


	vector<Rect> boundRect_lackLine(contours_lackline.size());
	{
		uint8_t Find_Flag = 0;

		vector<double> Area_list_Find;
		vector<float> W_list_Find;
		vector<float> H_list_Find;
		vector<int> X_list_Find;
		vector<int> Y_list_Find;
		vector<int> X_R_list_Find;
		vector<int> Y_R_list_Find;

		vector<double> Area_list_Maybe;
		vector<float> W_list_Maybe;
		vector<float> H_list_Maybe;
		vector<int> X_list_Maybe;
		vector<int> Y_list_Maybe;
		vector<int> X_R_list_Maybe;
		vector<int> Y_R_list_Maybe;
		vector<unsigned long long> Index_Find;
		vector<unsigned long long> Index_Maybe;
		for (vector<int>::size_type i = 0; i < contours_lackline.size(); i++)
		{
			//RotatedRect rect = minAreaRect(contours_lackline[i]);  //包覆轮廓的最小斜矩形
			//Point p = rect.center;
			////按图片坐标系为准，左上角原点,上侧Y轴，左侧X轴
			//X_list.push_back(p.y);
			//Y_list.push_back(p.x);
			double area = contourArea(contours_lackline[i]);
			if (area > 300)
			{
				/*Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
				drawContours(temp_mask, contours_lackline, i, 255, FILLED, 8);*/
				boundRect_lackLine[i] = boundingRect(Mat(contours_lackline[i]));
				float w = boundRect_lackLine[i].width;
				float h = boundRect_lackLine[i].height;
				int X_1 = boundRect_lackLine[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect_lackLine[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect_lackLine[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect_lackLine[i].br().y;//矩形右下角Y坐标值

				//求取偏心率
				double long_axis = max(w, h) / 2;
				double short_axis = min(w, h) / 2;
				double Eccentricity = sqrt(1 - (short_axis*short_axis / (long_axis*long_axis)));
				double longShortRatio = h / w;
				double Area_Ratio = area / (h*w);
				//求与X轴夹角
				double Line_angle = atan(longShortRatio) * 180 / (3.1415926);

				if (Eccentricity >= 0.95 && Line_angle <= 4.5 && w > 1000 && h > 20)
				{

					cv::drawContours(Marke_mask, contours_lackline, int(i), 255, FILLED, 8); //制作马克笔掩膜
					Area_list_Find.push_back(area);

					W_list_Find.push_back(w);
					H_list_Find.push_back(h);
					X_list_Find.push_back(X_1);
					Y_list_Find.push_back(Y_1);
					X_R_list_Find.push_back(X_2);
					Y_R_list_Find.push_back(Y_2);
					Index_Find.push_back(i);
					Find_Flag = Find_Flag + 1;
				}
				else if (Eccentricity >= 0.95 && Line_angle >= 4.5 && Line_angle <= 10 && Area_Ratio < 0.45) {
					Area_list_Maybe.push_back(area);
					X_list_Maybe.push_back(X_1);
					Y_list_Maybe.push_back(Y_1);
					W_list_Maybe.push_back(w);
					H_list_Maybe.push_back(h);
					X_R_list_Maybe.push_back(X_2);
					Y_R_list_Maybe.push_back(Y_2);
					Index_Maybe.push_back(i);
				}
				else if (Eccentricity >= 0.95 && Line_angle <= 4.5) {
					Area_list_Maybe.push_back(area);
					X_list_Maybe.push_back(X_1);
					Y_list_Maybe.push_back(Y_1);
					W_list_Maybe.push_back(w);
					H_list_Maybe.push_back(h);
					X_R_list_Maybe.push_back(X_2);
					Y_R_list_Maybe.push_back(Y_2);
					Index_Maybe.push_back(i);
				}
				else {
					continue;
				}
			}
		}
		if (F_strategy == 0) //策略1 修复
		{
			if (Find_Flag > 0)//二次遍历
			{
				uint8_t Sort_Num = Find_Flag;
				if (Find_Flag == 1)
				{
					for (vector<int>::size_type Index = 0; Index < X_list_Maybe.size(); Index++)
					{
						if (abs(Y_list_Maybe[Index] - Y_list_Find[0]) < 50)//寻找与原曲线在同一范围内的标注线
						{
							cv::drawContours(Marke_mask, contours_lackline, int(Index_Maybe[Index]), 255, FILLED, 8); //制作马克笔掩膜
							Area_list_Find.push_back(Area_list_Maybe[Index]);
							W_list_Find.push_back(W_list_Maybe[Index]);
							H_list_Find.push_back(H_list_Maybe[Index]);
							X_list_Find.push_back(X_list_Maybe[Index]);
							Y_list_Find.push_back(Y_list_Maybe[Index]);
							X_R_list_Find.push_back(X_R_list_Maybe[Index]);
							Y_R_list_Find.push_back(Y_R_list_Maybe[Index]);
							Index_Find.push_back(Index_Maybe[Index]);
							Find_Flag++;
						}
					}
				}
				else {
					for (uint8_t Find_Num = 0; Find_Num < Sort_Num; Find_Num++)
					{
						for (vector<int>::size_type Index = 0; Index < X_list_Maybe.size(); Index++)
						{
							if (abs(Y_list_Maybe[Index] - Y_list_Find[Find_Num] - H_list_Find[Find_Num] / 2) < 50)//寻找与原曲线在同一范围内的标注线
							{
								cv::drawContours(Marke_mask, contours_lackline, int(Index_Maybe[Index]), 255, FILLED, 8); //制作马克笔掩膜
								Area_list_Find.push_back(Area_list_Maybe[Index]);
								W_list_Find.push_back(W_list_Maybe[Index]);
								H_list_Find.push_back(H_list_Maybe[Index]);
								X_list_Find.push_back(X_list_Maybe[Index]);
								Y_list_Find.push_back(Y_list_Maybe[Index]);
								X_R_list_Find.push_back(X_R_list_Maybe[Index]);
								Y_R_list_Find.push_back(Y_R_list_Maybe[Index]);
								Index_Find.push_back(Index_Maybe[Index]);
								Find_Flag++;
							}
						}
					}
				}

				Mat Marke_Inpaint_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);                                  //生成空白掩膜图像

				//resultMat = Marke_mask.clone(); //修复前的二值图像

				bitwise_and(Marke_mask, img_gray, Marke_Inpaint_mask);

				Mat structure_element_Marke_Inpaint = getStructuringElement(MORPH_RECT, Size(5, 5));
				dilate(Marke_Inpaint_mask, Marke_Inpaint_mask, structure_element_Marke_Inpaint);

				//cv::imwrite("save.bmp", Marke_Inpaint_mask);
				Mat imgInpaint;
				cv::inpaint(img_gray, Marke_Inpaint_mask, imgInpaint, 5,INPAINT_NS); // 修复策略1

				inpaint(img_gray, Marke_Inpaint_mask, imgInpaint, 5, INPAINT_TELEA);// 修复策略2 测试该效果更好一些
				resultMat = imgInpaint.clone(); //修复后的灰度图像
			}
			else {

				resultMat = Mat::zeros(src.rows, src.cols, CV_8UC1);                                  //生成空白掩膜图像
			}
		}
		else { //策略2 给出区域不修复

			if (Find_Flag > 0)//二次遍历
			{
				if (Find_Flag == 1)
				{
					resultMat(Rect(0, Y_list_Find[0], src.cols, int(H_list_Find[0]))) = uchar(255);
				}
				else {
					int Min_num_L = *min_element(Y_list_Find.begin(), Y_list_Find.end()); //获取最小值
					int Min_num_L_2 = *min_element(Y_R_list_Find.begin(), Y_R_list_Find.end()); //获取最小值

					int Max_num_L = *max_element(Y_list_Find.begin(), Y_list_Find.end()); //获取最大值
					int Max_num_L_2 = *max_element(Y_R_list_Find.begin(), Y_R_list_Find.end()); //获取最大值

					resultMat(Rect(0, min(Min_num_L, Min_num_L_2), src.cols, max(Max_num_L, Max_num_L_2) - min(Min_num_L, Min_num_L_2))) = uchar(255);
				}
			}
			else {

				resultMat = Mat::zeros(src.rows, src.cols, CV_8UC1);                                  //生成空白掩膜图像

			}
		}

	}

	return resultMat;
}


/*************************************************
//  Method:    convertTo3Channels
//  Description: 将单通道图像转为三通道图像
//  Returns:   cv::Mat
//  Parameter: binImg 单通道图像对象
*************************************************/
Mat convertTo3Channels(const Mat& binImg)
{
	Mat three_channel = Mat::zeros(binImg.rows, binImg.cols, CV_8UC3);
	vector<Mat> channels;
	for (int i = 0; i < 3; i++)
	{
		channels.push_back(binImg);
	}
	merge(channels, three_channel);
	return three_channel;
}


bool spurtcode(Mat white_yiwu, Mat ceguang, Mat* mresult, string* causecolor) {
	bool result = false;
	Mat coderect = white_yiwu(Rect(1545, 678, 350, 550));
	double meanrect = 0.95*mean(coderect)[0];
	Mat out1;
	inRange(coderect, meanrect, 255, out1);
	threshold(coderect, out1, meanrect, 255, CV_THRESH_BINARY_INV);
	adaptiveThresholdCustom(coderect, out1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 45, 3, 1, true);
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
	Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(out1, out1, MORPH_CLOSE, element1);   //开运算形态学操作。可以减少噪点

	erode(out1, out1, element);

	out1 = ~out1;
	Mat test = out1 + coderect;
	Mat out2;
	inpaint(coderect, out1, out2, 9, INPAINT_TELEA);
	out2.copyTo(white_yiwu(Rect(1545, 678, 350, 550)));
	return result;


}
/*=========================================================
* 函 数 名: lack_line_gray
* 功能描述: 少线缺陷判断(灰底)
* 函数输入：主相机白底图像和主相机拍摄侧光图像
* 备注说明：2020年7月15日修改
=========================================================*/
bool lack_line_gray(Mat white_yiwu, Mat ceguang, Mat* mresult, string *causecolor, string name)
{
	bool firstresult = true;
	bool result = false;
	white_yiwu = gamma(white_yiwu, 0.5);
	Mat img_gray = white_yiwu.clone();
	Mat img_gray1 = white_yiwu.clone();
	//imwrite("D://img_gray.bmp",img_gray);
	Mat gray_ceguang = ceguang.clone();
	Mat gray_ceguang1 = ceguang.clone();
	Mat imageBinary;//分割二值图
	Mat ceimageBinary;//侧光分割二值图
	Mat ceimageBinary1;//分割二值图,用于测光排除
	//统计同一列中独立像素点个数
	vector<int> X_list;
	vector<int> Y_list;
	vector<int> count_x;
	vector<int> count_y;
	count_x.resize(img_gray.rows);
	count_y.resize(img_gray.cols);
	medianBlur(img_gray, img_gray, 3);
	//侧光滤波
	medianBlur(gray_ceguang, gray_ceguang, 9);
	medianBlur(gray_ceguang1, gray_ceguang1, 5);//用于侧光屏蔽
	//屏蔽边缘
	Mat shieldMask;
	Mat shieldMask2;
	int bored = 30;
	Mat shieldMask3 = Mat::zeros(img_gray.size(), CV_8UC1);
	threshold(img_gray(Rect(0, 0, 0.8*img_gray.cols, img_gray.rows)), shieldMask, 0.8 * mean(img_gray)[0], 255, CV_THRESH_BINARY);
	shieldMask(Rect(0, 0, shieldMask.cols, bored)) = 0;
	shieldMask(Rect(0, shieldMask.rows - bored, shieldMask.cols, bored)) = 0;
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(25, 25));
	erode(shieldMask, shieldMask, dilateElement);
	threshold(img_gray(Rect(0.8*img_gray.cols, 0, 0.2*img_gray.cols, img_gray.rows)), shieldMask2, 1.3 * mean(img_gray)[0], 255, CV_THRESH_BINARY_INV);
	shieldMask2(Rect(0, 0, shieldMask2.cols, bored)) = 0;
	shieldMask2(Rect(0, shieldMask2.rows - bored, shieldMask2.cols, bored)) = 0;
	erode(shieldMask2, shieldMask2, dilateElement);
	shieldMask.copyTo(shieldMask3(Rect(0, 0, 0.8*img_gray.cols, img_gray.rows)));
	shieldMask2.copyTo(shieldMask3(Rect(0.8*img_gray.cols, 0, 0.2*img_gray.cols, img_gray.rows)));



	adaptiveThresholdCustom(img_gray, imageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 21, 5, 1, 0.5);//21 5  LQY:17 3


	//侧光二值化
	adaptiveThresholdCustom(gray_ceguang, ceimageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 31, 3, 1, 0.5);//21 5  LQY:17 3
	adaptiveThresholdCustom(gray_ceguang1, ceimageBinary1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 3, 1, 0.5);//21 5  LQY:17 3   //31  3

	//点状线屏蔽
	Mat LSimageBinary = imageBinary(Rect(240, 310, 180, 180));
	Mat RXimageBinary = imageBinary(Rect(2580, 1000, 130, 120));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(11, 11));//闭操作结构元素
	//erode(LSimageBinary, LSimageBinary, element2);
	Mat LSimageBinaryTemp;
	Mat RXimageBinaryTemp;
	morphologyEx(LSimageBinary, LSimageBinaryTemp, MORPH_CLOSE, element2);
	morphologyEx(RXimageBinary, RXimageBinaryTemp, MORPH_CLOSE, element2);
	bitwise_and(~LSimageBinaryTemp, LSimageBinary, LSimageBinary);
	bitwise_and(~RXimageBinaryTemp, RXimageBinary, RXimageBinary);
	LSimageBinary.copyTo(imageBinary(Rect(240, 310, 180, 180)));
	RXimageBinary.copyTo(imageBinary(Rect(2580, 1000, 130, 120)));

	medianBlur(ceimageBinary, ceimageBinary, 17);

	imageBinary(Rect(imageBinary.cols - 250, 0, 250, 250)) = uchar(0);//去除易撕贴部分域
	imageBinary(Rect(0, 0, imageBinary.cols - 1, 20)) = uchar(0);                          //去除四个边界的影响上边 10
	imageBinary(Rect(0, imageBinary.rows - 10, imageBinary.cols - 1, 10)) = uchar(0);      //下边 12
	imageBinary(Rect(0, 0, 40, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	imageBinary(Rect(imageBinary.cols - 40, 0, 40, imageBinary.rows - 1)) = uchar(0);      //右边
	imageBinary(Rect(imageBinary.cols - 600, 500, 600, 500)) = uchar(0);
	//侧光屏蔽
	ceimageBinary(Rect(imageBinary.cols - 260, 0, 260, 250)) = uchar(0);//去除易撕贴部分域
	ceimageBinary(Rect(0, 0, imageBinary.cols - 1, 20)) = uchar(0);                          //去除四个边界的影响上边 10
	ceimageBinary(Rect(0, imageBinary.rows - 10, imageBinary.cols - 1, 10)) = uchar(0);      //下边 12
	ceimageBinary(Rect(0, 0, 20, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	ceimageBinary(Rect(imageBinary.cols - 30, 0, 30, imageBinary.rows - 1)) = uchar(0);      //右边
	ceimageBinary(Rect(imageBinary.cols - 300, 500, 300, 500)) = uchar(0);

	Mat element21 = getStructuringElement(MORPH_RECT, Size(7, 7));//闭操作结构元素
	dilate(ceimageBinary1, ceimageBinary1, element21);
	bitwise_and(~ceimageBinary1, imageBinary, imageBinary);        //侧光屏蔽

	vector<vector<Point>> contours_lackline;
	vector<vector<Point>> contours_lacklinece;
	//侧光轮廓排除
	findContours(ceimageBinary, contours_lacklinece, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);   //外围轮廓2022.2.8
	vector<Rect> boundRect_lackLinece(contours_lacklinece.size());
	Mat temp_maskce = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
	for (vector<int>::size_type i = 0; i < contours_lacklinece.size(); i++) {
		double area = contourArea(contours_lacklinece[i]);
		RotatedRect rect = minAreaRect(contours_lacklinece[i]);  //包覆轮廓的最小斜矩形
		double angle = rect.angle;
		if (area > 500 && (angle < -80 || angle >-10)) {
			Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
			drawContours(temp_mask, contours_lacklinece, i, 255, FILLED, 8);
			boundRect_lackLinece[i] = boundingRect(Mat(contours_lacklinece[i]));
			float w = boundRect_lackLinece[i].width;
			float h = boundRect_lackLinece[i].height;
			int X_1 = boundRect_lackLinece[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect_lackLinece[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect_lackLinece[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect_lackLinece[i].br().y;//矩形右下角Y坐标值
			double longShortRatio = max(h / w, w / h);
			double 	longShortlowerth = 10;//11 4
			int shortHigherth = 100;
			//if (longShortRatio > longShortlowerth && min(w, h) < shortHigherth && max(w, h) > 40)//长宽比，长度，宽度限制
			if (longShortRatio > 5 && min(w, h) < 250 && max(w, h) > 50)//长宽比，长度，宽度限制
			{
				drawContours(temp_maskce, contours_lacklinece, i, 255, FILLED, 8);
			}
		}

	}
	Mat structure_element2 = getStructuringElement(MORPH_RECT, Size(15, 15));
	dilate(temp_maskce, temp_maskce, structure_element2);
	bitwise_and(imageBinary, ~temp_maskce, imageBinary);
	bitwise_and(imageBinary, shieldMask3, imageBinary);

	//屏蔽

	return result;
}
bool lack_line(Mat white_yiwu, Mat ceguang, Mat Mask_MY, Mat* mresult, string *causecolor, string name)
{
	bool firstresult = true;
	bool result = false;
	Mat img_gray = white_yiwu.clone();
	Mat img_gray1 = white_yiwu.clone();
	//Mat src_white_yiwu_copy = src_white_yiwu.clone();
	Mat gray_ceguang = ceguang.clone();
	Mat gray_ceguang1 = ceguang.clone();
	Mat imageBinary;//分割二值图
	Mat imageBinaryce1;//分割二值图
	//掩膜处理去除刘海干扰
	Mat th1;
	threshold(img_gray, th1, 20, 255, CV_THRESH_BINARY);
	Mat structure_element2 = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(th1, th1, structure_element2);
	bitwise_and(img_gray, th1, img_gray);
	medianBlur(gray_ceguang, gray_ceguang, 9);
	medianBlur(gray_ceguang1, gray_ceguang1, 5);//用于侧光屏蔽
//    adaptiveThreshold(img_gray, imageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,SXPara->white_KernelSize, SXPara->white_GrayDifference);//lackline_bolckSize, lackline_delta
	adaptiveThreshold(img_gray, imageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 2);
	adaptiveThresholdCustom(gray_ceguang1, imageBinaryce1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 4, 1, 0.5);//21 5  LQY:17 3
	Mat element21 = getStructuringElement(MORPH_RECT, Size(7, 7));//闭操作结构元素
	dilate(imageBinaryce1, imageBinaryce1, element21);
	//    bitwise_and(~imageBinaryce1, imageBinary, imageBinary);        //侧光排除，

		//点状线屏蔽
	Mat leftCeyuan = imageBinary(Rect(270, 340, 200, 190));
	Mat rightCeyuan = imageBinary(Rect(2560, 1000, 190, 170));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(7, 7));//闭操作结构元素
	//erode(rightCeyuan, rightCeyuan, element2);
	Mat rightCeyuanTemp;
	Mat leftCeyuanTemp;
	morphologyEx(rightCeyuan, rightCeyuanTemp, MORPH_CLOSE, element2);
	morphologyEx(leftCeyuan, leftCeyuanTemp, MORPH_CLOSE, element2);
	bitwise_and(~leftCeyuanTemp, leftCeyuan, leftCeyuan);
	bitwise_and(~rightCeyuanTemp, rightCeyuan, rightCeyuan);
	rightCeyuan.copyTo(imageBinary(Rect(2560, 1000, 190, 170)));//2560 1000  160  140
	leftCeyuan.copyTo(imageBinary(Rect(270, 340, 200, 190)));

	imageBinary(Rect(imageBinary.cols - 250, 0, 250, 250)) = uchar(0);//去除易撕贴部分域
	//边缘
	imageBinary(Rect(0, imageBinary.rows - 30, imageBinary.cols, 30)) = uchar(0);      //下边
	imageBinary(Rect(0, 0, 40, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	imageBinary(Rect(imageBinary.cols - 40, 0, 40, imageBinary.rows - 1)) = uchar(0);      //右边
	imageBinary(Rect(0, 0, imageBinary.cols, 30)) = uchar(0);               //上边 10
	//R角
	imageBinary(Rect(0, 0, 100, 100)) = uchar(0);               //左上角
	imageBinary(Rect(imageBinary.cols - 100, 0, 100, 100)) = uchar(0);               //右上角
	imageBinary(Rect(0, imageBinary.rows - 100, 100, 100)) = uchar(0);               //左下角
	imageBinary(Rect(imageBinary.cols - 100, imageBinary.rows - 100, 100, 100)) = uchar(0);               //右下角

	bitwise_and(imageBinary, Mask_MY, imageBinary);
	//V 0.1 水平少线
	Mat imageBinary_Level_line = imageBinary.clone();
	Mat elementTest_Level = getStructuringElement(MORPH_RECT, Size(1, 11));
	morphologyEx(imageBinary_Level_line, imageBinary_Level_line, MORPH_CLOSE, elementTest_Level);
	morphologyEx(imageBinary_Level_line, imageBinary_Level_line, MORPH_OPEN, elementTest_Level);

	vector<vector<Point>> contours_lackline;
	Mat elementTest = getStructuringElement(MORPH_RECT, Size(11, 1));
	morphologyEx(imageBinary, imageBinary, MORPH_CLOSE, elementTest);
	morphologyEx(imageBinary, imageBinary, MORPH_OPEN, elementTest);
	//按轮廓长度排序
	
	findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);

	std::sort(contours_lackline.begin(), contours_lackline.end(), sortContoursLength);
	vector<Rect> boundRect_lackLine(contours_lackline.size());
	{

	}
	// V0.1 水平少线补丁 
	if (result == false)// 策略一、二未触发时 水平少线补丁生效
	{
		vector<vector<Point>> contours_lackline_Level;
		findContours(imageBinary_Level_line, contours_lackline_Level, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
		sort(contours_lackline_Level.begin(), contours_lackline_Level.end(), compareContourAreas);
		vector<Rect> boundRect_lackLine_Level(contours_lackline_Level.size());
		{
			for (vector<int>::size_type i = 0; i < contours_lackline_Level.size(); i++)
			{
				RotatedRect rect = minAreaRect(contours_lackline_Level[i]);  //包覆轮廓的最小斜矩形

				double angle = rect.angle;
				//cout << "angle" << angle;
				Point p = rect.center;
				//按图片坐标系为准，左上角原点,上侧Y轴，左侧X轴
				//X_list.push_back(p.y);
				//Y_list.push_back(p.x);

				double area = contourArea(contours_lackline_Level[i]);
				//cout<<"area:"<<area<<endl;
				if (area > 50 && area < 50000 && (angle < -85 || angle >-5))
				{
					Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
					drawContours(temp_mask, contours_lackline_Level, i, 255, FILLED, 8);
					boundRect_lackLine_Level[i] = boundingRect(Mat(contours_lackline_Level[i]));
					float w = boundRect_lackLine_Level[i].width;
					float h = boundRect_lackLine_Level[i].height;
					int X_1 = boundRect_lackLine_Level[i].tl().x;//矩形左上角X坐标值
					int Y_1 = boundRect_lackLine_Level[i].tl().y;//矩形左上角Y坐标值
					int X_2 = boundRect_lackLine_Level[i].br().x;//矩形右下角X坐标值
					int Y_2 = boundRect_lackLine_Level[i].br().y;//矩形右下角Y坐标值
					Moments m = moments(contours_lackline_Level[i]);
					int x_point = int(m.m10 / m.m00);
					int y_point = int(m.m01 / m.m00);
					if (x_point < 0 || x_point >= imageBinary.cols || y_point < 0 || y_point >= imageBinary.rows)
					{
						continue;
					}
					if (w > h) {
						continue;
					}
					//位置限定 排除易撕贴 刘海区域干扰
					//区域划分 起始位置
					bool Edge_Flag = 1;
					if (X_1> 2750 && (Y_2< 550 || Y_1 > 1000))
					{
						Edge_Flag = 0; //位于头部刘海区域 
					}
					if ( (X_1 + w > 2750 && Y_1 + h < 300) || ( X_1 + w > 2850 && ( Y_1>500 && Y_2<1000)) || ( Y_1 > 250 && Edge_Flag))
					{
						continue;
					}
					double longShortRatio = max(h / w, w / h);
					//cout << "longShortRatio:" << longShortRatio << " SXPara->white_LongShortlowerth:" << SXPara->white_LongShortlowerth << " min(w, h):" << min(w, h) << " SXPara->white_LengthMinHighLimit:" << SXPara->white_LengthMinHighLimit << " max(w, h):" << max(w, h) << " SXPara->white_LengthMaxLowLimit:" << SXPara->white_LengthMaxLowLimit << endl;
					int Line_Level_Length = 0;
					if (Edge_Flag == 0)
					{
						Line_Level_Length = 200;
					}
					else
					{
						Line_Level_Length = 550;
					}
					if (longShortRatio >= 5 && min(w, h) < 30 && max(w, h) > Line_Level_Length)//长宽比，长度，宽度限制  8 40 250
					{
						int border = 3;//选定框边界宽度
						int x_lt = X_1 - border;
						//越界保护
						if (x_lt < 0)
						{
							x_lt = 0;
						}
						int y_lt = Y_1 - border;
						if (y_lt < 0)
						{
							y_lt = 0;
						}
						int x_rt = X_2 + border;
						if (x_rt > img_gray.size[1] - 1)
						{
							x_rt = img_gray.size[1] - 1;
						}
						int y_rt = Y_2 + border;
						if (y_rt > img_gray.size[0] - 1)
						{
							y_rt = img_gray.size[0] - 1;
						}
						//侧光图排除明显划痕
						Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
						Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
						double meanOut;//缺陷外围灰度均值
						double meanIn;//缺陷区域灰度均值
						Mat maskGray;//做掩膜后的图像

						int grayValueSum = 0;//像素灰度总和
						int pixelsNum = 0;//像素点个数
						bitwise_and(sidelightSuspect, mask, maskGray);//侧光图做掩膜
						//计算侧光图上缺陷区域灰度均值
						for (int i = 0; i < maskGray.cols; i++)
						{
							for (int j = 0; j < maskGray.rows; j++)
							{
								if (maskGray.at<uchar>(j, i) > 0)
								{
									grayValueSum += maskGray.at<uchar>(j, i);
									pixelsNum++;
								}
							}
						}
						meanIn = grayValueSum / (float)pixelsNum;
						//计算侧光图上缺陷外围灰度均值
						grayValueSum = 0;
						pixelsNum = 0;
						bitwise_and(sidelightSuspect, ~mask, maskGray);
						for (int i = 0; i < maskGray.cols; i++)
						{
							for (int j = 0; j < maskGray.rows; j++)
							{
								if (maskGray.at<uchar>(j, i) <= meanIn && maskGray.at<uchar>(j, i) > 0)
								{
									grayValueSum += maskGray.at<uchar>(j, i);
									pixelsNum++;
								}
							}
						}
						meanOut = grayValueSum / (float)pixelsNum;
						double removeScratch = meanIn - meanOut;//缺陷区域与周围灰度差值(整体性)
						//计算方差
						cv::Mat meanGray;
						cv::Mat stdDev;
						for (int k = 0; k < sidelightSuspect.cols; k++)
						{
							for (int t = 0; t < sidelightSuspect.rows; t++)
							{
								if (sidelightSuspect.at<uchar>(t, k) == 0)
								{
									sidelightSuspect.at<uchar>(t, k) = meanOut;
								}
							}
						}
						cv::meanStdDev(sidelightSuspect, meanGray, stdDev);
						double sideLightStddev = stdDev.at<double>(0, 0);

						//加入白底参数进行缺陷排除
						Mat white_defect = img_gray(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//白底疑似缺陷区域图像
						double meanwhite_in = mean(white_defect, mask)[0];
						double meanwhite_out = mean(white_defect, ~mask)[0];
						double meanwhite_diff = meanwhite_in - meanwhite_out;

						double removeScratchFlag;
						if (area > 10000 && area < 20000)
							removeScratchFlag = 8;//8
						else if (area >= 20000)
							removeScratchFlag = 9;//9
						else
							removeScratchFlag = 7;//7
						if (removeScratch < removeScratchFlag && meanwhite_diff < -1.2)//lackscratchth 5 //sideLightStddev >3.2
						{
							//////样本生成 确定原始模板
							//Mat Temp_mask_en = white_defect.clone(); // 原始底图 一次Gabor滤波后的
							//Mat Temp_mask_en_End = convertTo3Channels(Temp_mask_en);
							////Mat Temp_mask_Bin = mask.clone();//二值化模板
							//Mat Temp_mask_Bin = 255 * cv::Mat::ones(Temp_mask_en.rows, Temp_mask_en.cols, Temp_mask_en.depth());
							////Mat Temp_mask_Bin = temp_mask.clone();//二值化模板
							//Mat Temp_Withe_Create = img_gray1.clone(); //生成图像
							//Mat Temp_Withe_Create_End = convertTo3Channels(Temp_Withe_Create);
							//Point center(Temp_mask_en.cols/2+250, Temp_mask_en.rows/2);
							//Mat normal_clone;
							//seamlessClone(Temp_mask_en_End, Temp_Withe_Create_End, Temp_mask_Bin, center, normal_clone, NORMAL_CLONE);

							//
								result = true;
								cout << "true--4" << endl;
								if (firstresult == true)
								{
									int defectType = 0;
									firstresult = false;
								}
								CvPoint top_lef4 = cvPoint(x_lt, y_lt);
								CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
								rectangle(img_gray, top_lef4, bottom_right4, Scalar(0, 255, 0), 5, 8, 0);
								string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
								putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
								break;
						}
					}
				}
			}
		}
	}
	if (result == true)
	{
		*mresult = img_gray;
		*causecolor = "少线";


		result = true;
	}
	return result;
}



/*=========================================================
* 函 数 名: lack_line_black(Mat white_yiwu, Mat ceguang, Mat* mresult, string* causecolor)
* 功能描述: 少线缺陷判断
* 函数输入：主相机白底图像和主相机拍摄侧光图像
* 备注说明：2020年9月18日修改
=========================================================*/
bool lack_line_black(Mat white_yiwu, Mat ceguang, Mat *mresult, string *causecolor, string name)
{
	bool result = false;

	Mat img_gray = white_yiwu.clone();
	Mat gray_ceguang = ceguang.clone();
	Mat imageBinary, imageBinary1;//分割二值图
	Mat th_result;
	//统计同一列中独立像素点个数
	vector<int> X_list;
	vector<int> Y_list;
	vector<int> count_x;
	vector<int> count_y;
	count_x.resize(1500);
	count_y.resize(3000);

	adaptiveThreshold(img_gray, imageBinary1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 45, -2);//lackline_bolckSize, lackline_delta
	adaptiveThresholdCustom(img_gray, imageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 45, -2, 1, 2);

	int leftWidth = 30;
	int rightWidth = 30;
	int topWidth = 30;
	int bottomWidth = 30;

	Mat leftPart, leftPartBinary, rightPart, rightPartBinary, topPart, topPartBinary, bottomPart, bottomPartBinary;
	leftPart = img_gray(Rect(0, 0, leftWidth, img_gray.rows));
	adaptiveThresholdCustom(leftPart, leftPartBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3, 1, 2);//块大小，阈值大小，偏移量，平均值倍数,偏移比率

	rightPart = img_gray(Rect(img_gray.cols - rightWidth - 1, 0, rightWidth, img_gray.rows));
	adaptiveThresholdCustom(rightPart, rightPartBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 31, -2, 1, 1.5);//块大小，阈值大小，偏移量，平均值倍数,偏移比率

	topPart = img_gray(Rect(0, 0, img_gray.cols, topWidth));
	adaptiveThresholdCustom(topPart, topPartBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3, 1, 2);//块大小，阈值大小，偏移量，平均值倍数,偏移比率

	bottomPart = img_gray(Rect(0, img_gray.rows - bottomWidth - 1, img_gray.cols, bottomWidth));
	adaptiveThresholdCustom(bottomPart, bottomPartBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3, 1, 2);//块大小，阈值大小，偏移量，平均值倍数,偏移比率


	 //二值图拼接
	Mat middlePart = imageBinary(Rect(leftWidth, topWidth, img_gray.cols - leftWidth - rightWidth, img_gray.rows - topWidth - bottomWidth));
	Mat imageBinary3 = Mat::zeros(imageBinary.size(), CV_8UC1);
	leftPartBinary.copyTo(imageBinary3(Rect(0, 0, leftWidth, imageBinary3.rows)));
	rightPartBinary.copyTo(imageBinary3(Rect(imageBinary3.cols - rightWidth - 1, 0, rightWidth, imageBinary3.rows)));
	topPartBinary.copyTo(imageBinary3(Rect(0, 0, imageBinary3.cols, topWidth)));
	bottomPartBinary.copyTo(imageBinary3(Rect(0, imageBinary3.rows - bottomWidth - 1, imageBinary3.cols, bottomWidth)));
	middlePart.copyTo(imageBinary3(Rect(leftWidth, topWidth, imageBinary3.cols - leftWidth - rightWidth, imageBinary3.rows - topWidth - bottomWidth)));


	Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));                            //开操作结构元素
	morphologyEx(imageBinary3, th_result, CV_MOP_OPEN, element);                              //先腐蚀，再膨胀，清除噪点
	Mat element1 = getStructuringElement(MORPH_RECT, Size(2, 2));                           //膨胀结构元素
	dilate(th_result, imageBinary3, element1);	                                                //扩扩大缺陷信息

	//imageBinary(Rect(imageBinary.cols -151, 0, 150, 150)) = uchar(0);                      //1017新加的   修改人：郭栋
	//imageBinary(Rect(imageBinary.cols - 200, imageBinary.rows - 200, 200, 200)) = uchar(0);//去除易撕贴部分
	//imageBinary(Rect(0, 0, imageBinary.cols - 1, 15)) = uchar(0);                          //去除四个边界的影响上边 10
	//imageBinary(Rect(0, imageBinary.rows - 12, imageBinary.cols - 1, 12)) = uchar(0);      //下边
	//imageBinary(Rect(0, 0, 22, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况 40
	//imageBinary(Rect(imageBinary.cols - 15, 0, 15, imageBinary.rows - 1)) = uchar(0);      //右边

	imageBinary(Rect(imageBinary.cols - 200, 0, 200, 250)) = uchar(0);//去除易撕贴部分域
	imageBinary(Rect(0, 0, imageBinary.cols - 1, 25)) = uchar(0);                          //去除四个边界的影响上边 10
	imageBinary(Rect(0, imageBinary.rows - 12, imageBinary.cols - 1, 12)) = uchar(0);      //下边 12
	imageBinary(Rect(0, 0, 60, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	imageBinary(Rect(imageBinary.cols - 80, 0, 80, imageBinary.rows - 1)) = uchar(0);      //右边15-18
	imageBinary(Rect(imageBinary.cols - 130, 525, 130, 440)) = uchar(0);

	imageBinary3(Rect(imageBinary.cols - 200, 0, 200, 250)) = uchar(0);//去除易撕贴部分域
	imageBinary3(Rect(0, 0, imageBinary.cols - 1, 25)) = uchar(0);                          //去除四个边界的影响上边 10
	imageBinary3(Rect(0, imageBinary.rows - 12, imageBinary.cols - 1, 12)) = uchar(0);      //下边 12
	imageBinary3(Rect(0, 0, 60, imageBinary.rows - 1)) = uchar(0);                          //左边有误分割情况     40
	imageBinary3(Rect(imageBinary.cols - 80, 0, 80, imageBinary.rows - 1)) = uchar(0);      //右边15-18
	imageBinary3(Rect(imageBinary.cols - 130, 525, 130, 440)) = uchar(0);

	vector<vector<Point>> contours_lackline;
	findContours(imageBinary3, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	//if (contours_lackline.size() > 1000 && contours_lackline.size() < 2500)//1500
	//{
	//	Mat structure_element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
	//	erode(imageBinary, imageBinary, structure_element2);    //将模板腐蚀一下,为了去除边界影响,否则相与过后会有白边
	//	contours_lackline.clear();
	//	findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	//}
	//else if (contours_lackline.size() >= 2500)
	//	return false;

	imwrite("D:\\SX\\" + name + "_imageBinary_black.bmp", imageBinary3);

	vector<Rect> boundRect_lackLine(contours_lackline.size());
	{
		for (vector<int>::size_type i = 0; i < contours_lackline.size(); i++)
		{
			RotatedRect rect = minAreaRect(contours_lackline[i]);  //包覆轮廓的最小斜矩形
			Point p = rect.center;
			//按图片坐标系为准，左上角原点,上侧Y轴，左侧X轴
			X_list.push_back(p.y);
			Y_list.push_back(p.x);
			double area = contourArea(contours_lackline[i]);
			if (area > black_Area_min && area < black_Area_max)
			{
				Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
				drawContours(temp_mask, contours_lackline, i, 255, FILLED, 8);
				boundRect_lackLine[i] = boundingRect(Mat(contours_lackline[i]));
				float w = boundRect_lackLine[i].width;
				float h = boundRect_lackLine[i].height;
				int X_1 = boundRect_lackLine[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect_lackLine[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect_lackLine[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect_lackLine[i].br().y;//矩形右下角Y坐标值
				Moments m = moments(contours_lackline[i]);
				int x_point = int(m.m10 / m.m00);
				int y_point = int(m.m01 / m.m00);
				if (x_point < 0 || x_point >= imageBinary.cols || y_point < 0 || y_point >= imageBinary.rows)
				{
					continue;
				}
				double longShortRatio = max(h / w, w / h);
				if (longShortRatio > black_LongShortlowerth && min(w, h) < black_lengthMinHighLimit && max(w, h) > black_lengthMaxLowLimit)//长宽比，长度，宽度限制  长宽比11-8，长度上限60-50        8->7   50 ->42
				{
					int border = 3;//选定侧光框边界宽度
					int x_lt = X_1 - border;
					//越界保护
					if (x_lt < 0)
					{
						x_lt = 0;
					}
					int y_lt = Y_1 - border;
					if (y_lt < 0)
					{
						y_lt = 0;
					}
					int x_rt = X_2 + border;
					if (x_rt > img_gray.size[1] - 1)
					{
						x_rt = img_gray.size[1] - 1;
					}
					int y_rt = Y_2 + border;
					if (y_rt > img_gray.size[0] - 1)
					{
						y_rt = img_gray.size[0] - 1;
					}
					//侧光图排除明显划痕
					Mat black_defect = img_gray(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//黑底疑似缺陷区域图像
					Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
					Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜					
					double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
					double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
					double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                        //排除侧光图贴膜划痕的参数
					double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
					//计算方差
					cv::Mat meanGray;
					cv::Mat stdDev;
					for (int k = 0; k < sidelightSuspect.cols; k++)
					{
						for (int t = 0; t < sidelightSuspect.rows; t++)
						{
							if (sidelightSuspect.at<uchar>(t, k) == 0)
							{
								sidelightSuspect.at<uchar>(t, k) = meanGrayout_Suspect;
							}
						}
					}
					cv::meanStdDev(sidelightSuspect, meanGray, stdDev);
					double sideLightStddev = stdDev.at<double>(0, 0);
					double meanblack_in = mean(black_defect, mask)[0];
					double meanblack_out = mean(black_defect, ~mask)[0];
					double meanblack_diff = meanblack_in - meanblack_out;
					double removeStvflag;
					cout << "black_meanblack_diff:" << meanblack_diff << endl;
					double removeScratchFlag;
					cout << "black_removeScratch:" << removeScratch << "  removeScratchArea:" << removeScratchArea << "  sideLightStddev:" << sideLightStddev << endl;
					cout << "black_min((x_rt - x_lt - 2) , (y_rt - y_lt - 2)):" << min((x_rt - x_lt - 2), (y_rt - y_lt - 2)) << endl;
					if (removeScratchArea > 10000) {
						removeScratchFlag = black_AeraremoveScratch_2;//1.31
						removeStvflag = 15;
					}
					else {
						removeScratchFlag = black_AeraremoveScratch_1;//0.877
						removeStvflag = 5;
					}

					if (removeScratch < removeScratchFlag && min((x_rt - x_lt - 2), (y_rt - y_lt - 2)) < black_LenthMin_HighLimit)//lackscratchth 2021.7.13改15
					{
						if (meanblack_diff > black_meanBlackDiffLowLimit && sideLightStddev < removeStvflag)   //2
						{
							result = true;
							CvPoint top_lef4 = cvPoint(x_lt, y_lt);
							CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
							rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 0), 3, 8, 0);
							string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
							putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
							imwrite("D:\\SX\\" + name + "_lackline_information_black.bmp", img_gray);
						}
					}
				}
			}
		}
	}
	stack<int> x_stack;
	stack<int> y_stack;
	if (result == false)
	{
		int min_x;
		int max_x;
		for (vector<int>::size_type i = 0; i < X_list.size(); i++)
		{
			int temp_x = X_list.at(i);
			int temp_y = Y_list.at(i);

			if (count_x.at(temp_x) > 26)
			{
				if (!(x_stack.empty()))
				{
					int temp = x_stack.top();
					if (temp != temp_x)
					{
						//x_stack.pop();
						x_stack.push(temp_x);
					}
				}
				else
					x_stack.push(temp_x);
			}
			else
			{
				count_x[temp_x]++;
			}
			if (count_y.at(temp_y) > 26)
			{
				if (!(y_stack.empty()))
				{
					int temp = y_stack.top();
					if (temp != temp_y)
					{
						//y_stack.pop();
						y_stack.push(temp_y);
					}
				}
				else
					y_stack.push(temp_y);
			}
			else
			{
				count_y[temp_y]++;
			}
		}
		Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
		if (!(x_stack.empty()))
		{
			int index = x_stack.top();
			x_stack.pop();

			min_x = Y_list.at(0);
			max_x = Y_list.at(0);
			for (vector<int>::size_type i = 0; i < X_list.size(); i++)
			{
				int temp_y = Y_list.at(i);
				if (X_list.at(i) == index)
				{

					if (temp_y < min_x)
					{
						min_x = temp_y;
					}
					if (temp_y > max_x)
					{
						max_x = temp_y;
					}
				}
			}
			temp_mask(Rect(min_x, index, max_x - min_x, 1)) = uchar(255);
			int X_1 = min_x;//矩形左上角X坐标值
			int Y_1 = index;//矩形左上角Y坐标值
			int X_2 = max_x;//矩形右下角X坐标值
			int Y_2 = index;//矩形右下角Y坐标值
			int border = 3;//选定框边界宽度
			int x_lt = X_1 - border;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border;
			if (x_rt > img_gray.size[1] - 1)
			{
				x_rt = img_gray.size[1] - 1;
			}
			int y_rt = Y_2 + border;
			if (y_rt > img_gray.size[0] - 1)
			{
				y_rt = img_gray.size[0] - 1;
			}
			//侧光图排除明显划痕
			Mat black_defect = img_gray(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//黑底疑似缺陷区域图像
			Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
			Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
			double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
			double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
			double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                   //排除侧光图贴膜划痕的参数
			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);

			double meanblack_in = mean(black_defect, mask)[0];
			double meanblack_out = mean(black_defect, ~mask)[0];
			double meanblack_diff = meanblack_in - meanblack_out;

			double removeScratchFlag;
			if (removeScratchArea > 10000)
				removeScratchFlag = black_xyDirect_removeScratchFlag_2;
			else
				removeScratchFlag = black_xyDirect_removeScratchFlag_1;
			cout << "black_removeScratch:" << removeScratch << "  removeScratchArea:" << removeScratchArea << endl;
			if (removeScratch < removeScratchFlag)  //lackscratchth
			{

				result = true;
				CvPoint top_lef4 = cvPoint(x_lt, y_lt);
				CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
				rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
				string information = "th:" + to_string(removeScratch) + " area " + to_string(removeScratchArea);// +" width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
				putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
				imwrite("D:\\SX\\" + name + "_lackline_black.bmp", img_gray);
			}
		}

		if (!(y_stack.empty()) && result == false)
		{
			int index = y_stack.top();
			y_stack.pop();

			min_x = X_list.at(0);
			max_x = X_list.at(0);
			for (vector<int>::size_type i = 0; i < X_list.size(); i++)
			{
				int temp_x = X_list.at(i);
				if (Y_list.at(i) == index)
				{

					if (temp_x < min_x)
					{
						min_x = temp_x;
					}
					if (temp_x > max_x)
					{
						max_x = temp_x;
					}
				}
			}
			temp_mask(Rect(index, min_x, 1, max_x - min_x)) = uchar(255);
			int X_1 = index;//矩形左上角X坐标值
			int Y_1 = min_x;//矩形左上角Y坐标值
			int X_2 = index;//矩形右下角X坐标值
			int Y_2 = max_x;//矩形右下角Y坐标值
			int border = 3;//选定框边界宽度
			int x_lt = X_1 - border;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border;
			if (x_rt > img_gray.size[1] - 1)
			{
				x_rt = img_gray.size[1] - 1;
			}
			int y_rt = Y_2 + border;
			if (y_rt > img_gray.size[0] - 1)
			{
				y_rt = img_gray.size[0] - 1;
			}
			//侧光图排除明显划痕
			Mat black_defect = img_gray(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//黑底疑似缺陷区域图像
			Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
			Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
			double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
			double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
			double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                   //排除侧光图贴膜划痕的参数
			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);

			double meanblack_in = mean(black_defect, mask)[0];
			double meanblack_out = mean(black_defect, ~mask)[0];
			double meanblack_diff = meanblack_in - meanblack_out;
			double removeScratchFlag;
			if (removeScratchArea > 10000)
				removeScratchFlag = black_xyDirect_removeScratchFlag_2;
			else
				removeScratchFlag = black_xyDirect_removeScratchFlag_1;//3-0.12
			cout << "black_removeScratch:" << removeScratch << "  removeScratchArea:" << removeScratchArea << "  sideLightStddev:" << meanblack_diff << endl;
			if (abs(removeScratch) < removeScratchFlag)  //lackscratchth
			{
				if (meanblack_diff > black_meanBlackDiffLowLimit)
				{
					result = true;
					CvPoint top_lef4 = cvPoint(x_lt, y_lt);
					CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
					rectangle(img_gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
					string information = "th:" + to_string(removeScratch) + " area " + to_string(removeScratchArea);// +" width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
					putText(img_gray, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
					imwrite("D:\\SX\\" + name + "_lackline_black.bmp", img_gray);
				}


			}
		}
	}
	if (result == true)
	{
		//fprintf(pOutFile, "%s,", "黑底下");
		*mresult = img_gray;
		*causecolor = "少线";
		result = true;
	}
	return result;
}


/*=========================================================
* 函 数 名: lack_line_red
* 功能描述: 少线缺陷判断(红色)
* 函数输入：主相机白底图像和主相机拍摄侧光图像
* 备注说明：2021年5月12日修改
=========================================================*/
bool lack_line_red(Mat src_red, Mat ceguang, Mat* mresult, string* causecolor)
{

	bool result = false;
	Mat image_red = src_red.clone();
	medianBlur(image_red, image_red, 9);
	medianBlur(ceguang, ceguang, 9);
	Mat gray_ceguang = ceguang.clone();

	Mat imageBinaryblue;//分割二值图
	Mat imageBinarygreen;//分割二值图
	Mat imageBinaryred;//分割二值图
	Mat imageBinaryblue2;//分割二值图
	Mat imageBinarygreen2;//分割二值图
	Mat imageBinaryred2;//分割二值图
	Mat ceimageBinary;//分割二值图
	//
	Mat imageBinary = Mat::zeros(src_red.size(), CV_8UC1);
	Mat imageBinary2 = Mat::zeros(src_red.size(), CV_8UC1);

	Mat shieldMask;
	Mat shieldMask2;
	int bored = 30;
	Mat shieldMask3 = Mat::zeros(src_red.size(), CV_8UC1);
	threshold(src_red(Rect(0, 0, 2500, 1500)), shieldMask, 0.8 * mean(src_red)[0], 255, CV_THRESH_BINARY);
	shieldMask(Rect(0, 0, shieldMask.cols, bored)) = 0;
	shieldMask(Rect(0, shieldMask.rows - bored, shieldMask.cols, bored)) = 0;
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(25, 25));
	erode(shieldMask, shieldMask, dilateElement);
	threshold(src_red(Rect(2500, 0, 500, 1500)), shieldMask2, 1.3 * mean(src_red)[0], 255, CV_THRESH_BINARY_INV);
	shieldMask2(Rect(0, 0, shieldMask2.cols, bored)) = 0;
	shieldMask2(Rect(0, shieldMask2.rows - bored, shieldMask2.cols, bored)) = 0;
	erode(shieldMask2, shieldMask2, dilateElement);
	shieldMask.copyTo(shieldMask3(Rect(0, 0, 2500, 1500)));
	shieldMask2.copyTo(shieldMask3(Rect(2500, 0, 500, 1500)));
	Mat blue = src_red(Rect(0, 0, 1000, src_red.rows)).clone();
	Mat green = src_red(Rect(1000, 0, 1000, src_red.rows)).clone();
	Mat red = src_red(Rect(2000, 0, 1000, src_red.rows)).clone();
	//normalize(blue, blue, 0, 255, CV_MINMAX);
	//convertScaleAbs(blue, blue);
	adaptiveThresholdCustom(blue, imageBinaryblue, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 2, 1, true);
	adaptiveThresholdCustom(green, imageBinarygreen, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 2, 1, true);
	adaptiveThresholdCustom(red, imageBinaryred, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 2, 1, true);
	adaptiveThresholdCustom(blue, imageBinaryblue2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 35, -2, 1, true);
	adaptiveThresholdCustom(green, imageBinarygreen2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 35, -2, 1, true);
	adaptiveThresholdCustom(red, imageBinaryred2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 35, -1.1, 1, true);

	imageBinaryblue(Rect(0, 0, 40, imageBinaryblue.rows)) = uchar(0);
	imageBinaryblue(Rect(imageBinaryblue.cols - 30, 0, 30, imageBinaryblue.rows)) = uchar(0);
	imageBinarygreen(Rect(0, 0, 30, imageBinarygreen.rows)) = uchar(0);
	imageBinarygreen(Rect(imageBinarygreen.cols - 30, 0, 30, imageBinarygreen.rows)) = uchar(0);
	imageBinaryred(Rect(0, 0, 30, imageBinaryred.rows)) = uchar(0);
	imageBinaryred(Rect(imageBinaryred.cols - 30, 0, 30, imageBinaryred.rows)) = uchar(0);
	imageBinaryblue.copyTo(imageBinary(Rect(0, 0, 1000, src_red.rows)));
	imageBinarygreen.copyTo(imageBinary(Rect(1000, 0, 1000, src_red.rows)));
	imageBinaryred.copyTo(imageBinary(Rect(2000, 0, 1000, src_red.rows)));

	imageBinaryblue2(Rect(0, 0, 40, imageBinaryblue.rows)) = uchar(0);
	imageBinaryblue2(Rect(imageBinaryblue.cols - 30, 0, 30, imageBinaryblue.rows)) = uchar(0);
	imageBinarygreen2(Rect(0, 0, 30, imageBinarygreen.rows)) = uchar(0);
	imageBinarygreen2(Rect(imageBinarygreen.cols - 30, 0, 30, imageBinarygreen.rows)) = uchar(0);
	imageBinaryred2(Rect(0, 0, 30, imageBinaryred.rows)) = uchar(0);
	imageBinaryred2(Rect(imageBinaryred.cols - 30, 0, 30, imageBinaryred.rows)) = uchar(0);
	imageBinaryblue2.copyTo(imageBinary2(Rect(0, 0, 1000, src_red.rows)));
	imageBinarygreen2.copyTo(imageBinary2(Rect(1000, 0, 1000, src_red.rows)));
	imageBinaryred2.copyTo(imageBinary2(Rect(2000, 0, 1000, src_red.rows)));

	bitwise_and(imageBinary, shieldMask3, imageBinary);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(imageBinary, imageBinary, MORPH_CLOSE, element);   //开运算形态学操作。可以减少噪点
	medianBlur(imageBinary, imageBinary, 5);

	bitwise_and(imageBinary2, shieldMask3, imageBinary2);
	morphologyEx(imageBinary2, imageBinary2, MORPH_CLOSE, element);   //开运算形态学操作。可以减少噪点
	medianBlur(imageBinary2, imageBinary2, 5);
	bitwise_or(imageBinary2, imageBinary, imageBinary);
	Mat element21 = getStructuringElement(MORPH_RECT, Size(55, 55));//闭操作结构元素

	Mat ceimageBinary2;
	adaptiveThresholdCustom(gray_ceguang, ceimageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, 2, 1, true);//21 5  LQY:17 3

	//adaptiveThresholdCustom(gray_ceguang, imageBinaryce, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 33,-3 , 1, 0.5);//21 5  LQY:17 3
	//morphologyEx(imageBinaryce, imageBinaryce, MORPH_OPEN, element21);   //闭运算形态学操作。可以减少噪点
	vector<vector<Point>> contours_lacklinece;

	findContours(ceimageBinary, contours_lacklinece, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);   //外围轮廓2022.2.8
	vector<Rect> boundRect_lackLinece(contours_lacklinece.size());
	Mat temp_maskce = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);

	for (vector<int>::size_type i = 0; i < contours_lacklinece.size(); i++) {
		double area = contourArea(contours_lacklinece[i]);

		RotatedRect rect = minAreaRect(contours_lacklinece[i]);  //包覆轮廓的最小斜矩形
		if (area > 500) {
			Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
			drawContours(temp_mask, contours_lacklinece, i, 255, FILLED, 8);
			boundRect_lackLinece[i] = boundingRect(Mat(contours_lacklinece[i]));
			float w = boundRect_lackLinece[i].width;
			float h = boundRect_lackLinece[i].height;
			int X_1 = boundRect_lackLinece[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect_lackLinece[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect_lackLinece[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect_lackLinece[i].br().y;//矩形右下角Y坐标值
			double longShortRatio = max(h / w, w / h);
			double 	longShortlowerth = 10;//11 4
			int shortHigherth = 100;
			//if (longShortRatio > longShortlowerth && min(w, h) < shortHigherth && max(w, h) > 40)//长宽比，长度，宽度限制
			if (longShortRatio > 3 && min(w, h) < 250 && max(w, h) > 50)//长宽比，长度，宽度限制
			{
				drawContours(temp_maskce, contours_lacklinece, i, 255, FILLED, 8);
			}
		}

	}
	Mat temp_maskce2 = temp_maskce(Rect(100, 100, temp_maskce.cols - 200, temp_maskce.rows - 200));
	dilate(temp_maskce2, temp_maskce2, element21);
	temp_maskce2.copyTo(ceimageBinary(Rect(100, 100, temp_maskce.cols - 200, temp_maskce.rows - 200)));
	//dilate(ceimageBinary, ceimageBinary, element21);
	bitwise_and(~ceimageBinary, imageBinary, imageBinary);

	imageBinary(Rect(imageBinary.cols - 130, 525, 130, 440)) = uchar(0);
	vector<vector<Point>> contours_lackline;
	findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours_lackline.begin(), contours_lackline.end(), compareContourAreas);
	if (contours_lackline.size() > 1000 && contours_lackline.size() < 6600)//1500 5000
	{
		Mat structure_element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(imageBinary, imageBinary, structure_element2);    //将模板腐蚀一下,为了去除边界影响,否则相与过后会有白边
		contours_lackline.clear();
		findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	}
	else if (contours_lackline.size() >= 6600)
		return false;
	vector<Rect> boundRect_lackLine(contours_lackline.size());
	{
		for (vector<int>::size_type i = 0; i < contours_lackline.size(); i++)
		{
			RotatedRect rect = minAreaRect(contours_lackline[i]);  //包覆轮廓的最小斜矩形
			Point p = rect.center;
			double angle = rect.angle;

			cout << "angle:" << angle << endl;  //&& angle<-80
			double area = contourArea(contours_lackline[i]);
			if (area > 300 && area < 50000 && (angle < -80 || angle >-10))
			{
				Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
				drawContours(temp_mask, contours_lackline, i, 255, FILLED, 8);
				boundRect_lackLine[i] = boundingRect(Mat(contours_lackline[i]));
				float w = boundRect_lackLine[i].width;
				float h = boundRect_lackLine[i].height;
				int X_1 = boundRect_lackLine[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect_lackLine[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect_lackLine[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect_lackLine[i].br().y;//矩形右下角Y坐标值
				Moments m = moments(contours_lackline[i]);
				int x_point = int(m.m10 / m.m00);
				int y_point = int(m.m01 / m.m00);
				if (x_point < 0 || x_point >= imageBinary.cols || y_point < 0 || y_point >= imageBinary.rows)
				{
					continue;
				}
				double longShortRatio = max(h / w, w / h);
				if (longShortRatio >= 6 && min(w, h) < 25 && max(w, h) > 250)//长宽比，长度，宽度限制
				{
					int border = 3;//选定框边界宽度
					int x_lt = X_1 - border;
					//越界保护
					if (x_lt < 0)
					{
						x_lt = 0;
					}
					int y_lt = Y_1 - border;
					if (y_lt < 0)
					{
						y_lt = 0;
					}
					int x_rt = X_2 + border;
					if (x_rt > src_red.size[1] - 1)
					{
						x_rt = src_red.size[1] - 1;
					}
					int y_rt = Y_2 + border;
					if (y_rt > src_red.size[0] - 1)
					{
						y_rt = src_red.size[0] - 1;
					}
					//侧光图排除明显划痕
					Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
					Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜

					double meanOut;//缺陷外围灰度均值
					double meanIn;//缺陷区域灰度均值
					Mat maskGray;//做掩膜后的图像

					int grayValueSum = 0;//像素灰度总和
					int pixelsNum = 0;//像素点个数
					bitwise_and(sidelightSuspect, mask, maskGray);//侧光图做掩膜
					//计算侧光图上缺陷区域灰度均值
					for (int i = 0; i < maskGray.cols; i++)
					{
						for (int j = 0; j < maskGray.rows; j++)
						{
							if (maskGray.at<uchar>(j, i) > 0)
							{
								grayValueSum += maskGray.at<uchar>(j, i);
								pixelsNum++;
							}
						}
					}
					meanIn = grayValueSum / (float)pixelsNum;

					//计算侧光图上缺陷外围灰度均值
					grayValueSum = 0;
					pixelsNum = 0;
					bitwise_and(sidelightSuspect, ~mask, maskGray);
					for (int i = 0; i < maskGray.cols; i++)
					{
						for (int j = 0; j < maskGray.rows; j++)
						{
							if (maskGray.at<uchar>(j, i) <= meanIn && maskGray.at<uchar>(j, i) > 0)
							{
								grayValueSum += maskGray.at<uchar>(j, i);
								pixelsNum++;
							}
						}
					}
					meanOut = grayValueSum / (float)pixelsNum;
					double removeScratch = meanIn - meanOut;//缺陷区域与周围灰度差值(整体性)
					//计算方差
					cv::Mat meanGray;
					cv::Mat stdDev;
					for (int k = 0; k < sidelightSuspect.cols; k++)
					{
						for (int t = 0; t < sidelightSuspect.rows; t++)
						{
							if (sidelightSuspect.at<uchar>(t, k) == 0)
							{
								sidelightSuspect.at<uchar>(t, k) = meanOut;
							}
						}
					}
					cv::meanStdDev(sidelightSuspect, meanGray, stdDev);
					double sideLightStddev = stdDev.at<double>(0, 0);
					Mat white_defect = src_red(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//白底疑似缺陷区域图像
					double meanwhite_in = mean(white_defect, mask)[0];
					double meanwhite_out = mean(white_defect, ~mask)[0];
					double meanwhite_diff = meanwhite_in - meanwhite_out;


					double removeScratchFlag;
					if (area > 10000 && area < 20000)
						removeScratchFlag = white_AeraremoveScratch_2;//1.3    5  6.5 7
					else if (area >= 20000)
						removeScratchFlag = white_AeraremoveScratch_3;
					else
						removeScratchFlag = white_AeraremoveScratch_1;//1.05  3

					if (removeScratch < removeScratchFlag  && abs(meanwhite_diff) >2)//lackscratchth 5 //sideLightStddev >3.2
					{
						result = true;
						CvPoint top_lef4 = cvPoint(x_lt, y_lt);
						CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
						rectangle(src_red, top_lef4, bottom_right4, Scalar(0, 255, 0), 5, 8, 0);
						string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
						putText(src_red, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
						*mresult = src_red;
						*causecolor = "少线";
						break;

						//imwrite("D:\\lackline-information.bmp", img_gray);
					}
				}
			}
		}
	}




	return result;


}

bool lack_line_rgb(Mat src_red, Mat ceguang, Mat* mresult, string* causecolor, string name) {
	bool result = false;
	Mat image_red = src_red.clone();
	medianBlur(image_red, image_red, 9);
	medianBlur(ceguang, ceguang, 9);

	Mat gray_ceguang = ceguang.clone();
	Mat cetest;
	adaptiveThresholdCustom(gray_ceguang, cetest, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 2, 1, true);
	Mat imageBinaryblue;//分割二值图
	Mat imageBinarygreen;//分割二值图
	Mat imageBinaryred;//分割二值图
	Mat imageBinaryblue2;//分割二值图
	Mat imageBinarygreen2;//分割二值图
	Mat imageBinaryred2;//分割二值图
	Mat ceimageBinary;//分割二值图
	int dividlinebg = 0;  //分割线
	int dividlinegr = 0;

	Mat imageBinary = Mat::zeros(src_red.size(), CV_8UC1);
	Mat imageBinary2 = Mat::zeros(src_red.size(), CV_8UC1);
	int ratio = 3;
	int kernel_size = 3;
	int lowThreshold = 5;
	//RGB区域划分
	Canny(image_red, imageBinary2, lowThreshold, lowThreshold * ratio, kernel_size); //边缘检测
	imageBinary2(Rect(0, 0, imageBinary2.cols, 100)) = 0;
	imageBinary2(Rect(0, imageBinary2.rows - 100, imageBinary2.cols, 100)) = 0;//上下屏蔽
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(imageBinary2, imageBinary2, MORPH_CLOSE, element);   //开运算形态学操作。可以减少噪点
	Vec4f leftLine_Fit, rightLine_Fit;				    //左右拟合直线数据
	bitwise_and(image_red, ~imageBinary2, imageBinary);
	vector<vector<Point>> contours_lacklinebg, contours_lacklinegr;
	Mat img1 = imageBinary2(Rect(900, 0, 200, imageBinary2.rows));
	findContours(imageBinary2(Rect(900, 0, 200, imageBinary2.rows)), contours_lacklinebg, CV_RETR_LIST, CHAIN_APPROX_NONE);   //外围轮廓2022.6.13
	sort(contours_lacklinebg.begin(), contours_lacklinebg.end(), compareContourSizes);
	if (contours_lacklinebg.size() != 0) {
		fitLine(contours_lacklinebg[0], leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
		cv::Point point0;
		point0.x = leftLine_Fit[2];
		point0.y = leftLine_Fit[3];
		cv::Point point1, point2;
		double k = leftLine_Fit[1] / leftLine_Fit[0];
		int x = (750.0 - point0.y) / k + point0.x;
		point1.x = 0;
		point1.y = k * (0 - point0.x) + point0.y;
		point2.x = 150;
		point2.y = k * (150 - point0.x) + point0.y;

		cv::line(img1, point1, point2, cv::Scalar(255, 255, 255), 2, 8, 0);
		/*int x_num=0;
		int n = 0;
		for (int i = 0; i < contours_lacklinebg[0].size(); i++) {

			x_num = x_num + contours_lacklinebg[0][i].x;
			n++;
		}
		x_num = x_num / n;*/

		dividlinebg = 900 + point0.x;  //偏移量获取
	}
	else {
		dividlinebg = 1000;
	}
	findContours(imageBinary2(Rect(1900, 0, 200, imageBinary2.rows)), contours_lacklinegr, CV_RETR_LIST, CHAIN_APPROX_NONE);   //外围轮廓2022.6.13
	sort(contours_lacklinegr.begin(), contours_lacklinegr.end(), compareContourAreas);
	if (contours_lacklinegr.size() != 0) {
		fitLine(contours_lacklinegr[0], rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //右侧拟合直线
		dividlinegr = 1900 + rightLine_Fit[2];
	}
	else {
		dividlinegr = 2000;
	}
	//区域提取
	Mat blue = src_red(Rect(0, 0, dividlinebg, src_red.rows)).clone();
	Mat green = src_red(Rect(dividlinebg, 0, dividlinegr - dividlinebg, src_red.rows)).clone();
	Mat red = src_red(Rect(dividlinegr, 0, src_red.cols - dividlinegr, src_red.rows)).clone();
	Mat shieldMask;
	Mat shieldMask2;
	Mat img_blue, img_green, img_red;
	Mat element1 = getStructuringElement(MORPH_RECT, Size(7, 7));
	double meanblue = mean(blue(Rect(0, 120, blue.cols, blue.rows - 240)))[0]; //求中间部分的平均灰度
	//Contrast(blue, -0.6, meanblue);

	threshold(red, shieldMask, 150, 255, CV_THRESH_BINARY);
	dilate(shieldMask, shieldMask, element1); //膨胀
	bitwise_and(red, ~shieldMask, red);       //将明显发分割线去掉
	Ptr<CLAHE> clahe = createCLAHE(1.8, Size(40, 40)); //实例化自适应直方图均衡化函数
	//clahe->apply(blue, img_blue);   //整图增强
	clahe->apply(green, img_green);   //整图增强
	clahe->apply(red, img_red);       //整图增强

	medianBlur(blue, img_blue, 3);
	medianBlur(green, img_green, 5);
	/*normalize(red, red, 0, 255, CV_MINMAX);
	convertScaleAbs(red, red);*/
	medianBlur(red, img_red, 3);
	//正负阈值分割
	adaptiveThresholdCustom(blue, imageBinaryblue, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 2, 1, true);
	adaptiveThresholdCustom(img_green, imageBinarygreen, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 3.5, 1, true);
	adaptiveThresholdCustom(img_red, imageBinaryred, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 35, 2.0, 1, true);
	adaptiveThresholdCustom(img_blue, imageBinaryblue2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 35, -2, 1, true);
	adaptiveThresholdCustom(img_green, imageBinarygreen2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 35, -3.5, 1, true);
	adaptiveThresholdCustom(img_red, imageBinaryred2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 35, -2.1, 1, true);
	morphologyEx(imageBinaryblue2, imageBinaryblue2, MORPH_CLOSE, element1);   //开运算形态学操作。可以减少噪点

   // 边缘屏蔽
	imageBinaryblue(Rect(0, 0, 40, imageBinaryblue.rows)) = uchar(0); //左
	imageBinaryblue(Rect(imageBinaryblue.cols - 30, 0, 30, imageBinaryblue.rows)) = uchar(0);//右
	imageBinarygreen(Rect(0, 0, 30, imageBinarygreen.rows)) = uchar(0);//左
	imageBinarygreen(Rect(imageBinarygreen.cols - 30, 0, 30, imageBinarygreen.rows)) = uchar(0);//右
	imageBinaryred(Rect(0, 0, 30, imageBinaryred.rows)) = uchar(0);
	imageBinaryred(Rect(imageBinaryred.cols - 30, 0, 30, imageBinaryred.rows)) = uchar(0);
	imageBinaryblue.copyTo(imageBinary(Rect(0, 0, dividlinebg, src_red.rows)));
	imageBinarygreen.copyTo(imageBinary(Rect(dividlinebg, 0, dividlinegr - dividlinebg, src_red.rows)));
	imageBinaryred.copyTo(imageBinary(Rect(dividlinegr, 0, src_red.cols - dividlinegr, src_red.rows))); //imageBinary正

	imageBinaryblue2(Rect(0, 0, 40, imageBinaryblue2.rows)) = uchar(0);
	imageBinaryblue2(Rect(imageBinaryblue2.cols - 30, 0, 30, imageBinaryblue2.rows)) = uchar(0);
	imageBinarygreen2(Rect(0, 0, 30, imageBinarygreen2.rows)) = uchar(0);
	imageBinarygreen2(Rect(imageBinarygreen2.cols - 30, 0, 30, imageBinarygreen2.rows)) = uchar(0);
	imageBinaryred2(Rect(0, 0, 30, imageBinaryred2.rows)) = uchar(0);
	imageBinaryred2(Rect(imageBinaryred2.cols - 30, 0, 30, imageBinaryred2.rows)) = uchar(0);
	imageBinaryblue2.copyTo(imageBinary2(Rect(0, 0, dividlinebg, src_red.rows)));
	imageBinarygreen2.copyTo(imageBinary2(Rect(dividlinebg, 0, dividlinegr - dividlinebg, src_red.rows)));
	imageBinaryred2.copyTo(imageBinary2(Rect(dividlinegr, 0, src_red.cols - dividlinegr, src_red.rows)));//imageBinary2
	imageBinary2(Rect(0, 0, imageBinary.cols, 20)) = uchar(0); //上屏蔽
	imageBinary2(Rect(0, imageBinary.rows - 20, imageBinary.cols, 20)) = uchar(0); //下屏蔽
	bitwise_or(imageBinary, imageBinary2, imageBinary2);
	//erode(imageBinary, imageBinary, element);
	morphologyEx(imageBinary, imageBinary, MORPH_OPEN, element);   //开运算形态学操作。可以减少噪点
	vector<vector<Point>> contours_lacklinefit;
	medianBlur(imageBinary, imageBinary, 3);  //得到imageBinary，imageBinary2

	//统计同一列中独立像素点个数
	vector<int> X_list;
	vector<int> Y_list;
	vector<int> count_x;
	vector<int> count_y;
	count_x.resize(imageBinary.rows);
	count_y.resize(imageBinary.cols);

	vector<vector<Point>> contours_lackline;
	findContours(imageBinary, contours_lackline, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours_lackline.begin(), contours_lackline.end(), compareContourAreas);
	vector<Rect> boundRect_lackLine(contours_lackline.size());

	imwrite("D:\\SX\\" + name + "_imageBinary_bgr.bmp", imageBinary);

	for (vector<int>::size_type i = 0; i < contours_lackline.size(); i++)
	{
		RotatedRect rect = minAreaRect(contours_lackline[i]);  //包覆轮廓的最小斜矩形
		Point p = rect.center;
		double angle = rect.angle;
		double area = contourArea(contours_lackline[i]);
		double width1 = min(rect.size.height, rect.size.width);
		if (width1 < 35 && area > 100 && area < 500000 && (angle < -80 || angle >-10))
		{
			//按图片坐标系为准，左上角原点,上侧Y轴，左侧X轴
			X_list.push_back(p.y);
			Y_list.push_back(p.x);
			Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
			drawContours(temp_mask, contours_lackline, i, 255, FILLED, 8);
			boundRect_lackLine[i] = boundingRect(Mat(contours_lackline[i]));
			float w = boundRect_lackLine[i].width;
			float h = boundRect_lackLine[i].height;
			int X_1 = boundRect_lackLine[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect_lackLine[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect_lackLine[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect_lackLine[i].br().y;//矩形右下角Y坐标值
			Moments m = moments(contours_lackline[i]);
			int x_point = int(m.m10 / m.m00);
			int y_point = int(m.m01 / m.m00);
			if (x_point < 0 || x_point >= imageBinary.cols || y_point < 0 || y_point >= imageBinary.rows)
			{
				continue;
			}
			double longShortRatio = max(h / w, w / h);
			if (longShortRatio >= 2 && min(w, h) < 25 && max(w, h) > 50)//长宽比，长度，宽度限制
			{
				int border = 3;//选定框边界宽度
				int x_lt = X_1 - border;
				//越界保护
				if (x_lt < 0)
				{
					x_lt = 0;
				}
				int y_lt = Y_1 - border;
				if (y_lt < 0)
				{
					y_lt = 0;
				}
				int x_rt = X_2 + border;
				if (x_rt > src_red.size[1] - 1)
				{
					x_rt = src_red.size[1] - 1;
				}
				int y_rt = Y_2 + border;
				if (y_rt > src_red.size[0] - 1)
				{
					y_rt = src_red.size[0] - 1;
				}
				//侧光图排除明显划痕
				Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
				Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜

				double meanOut;//缺陷外围灰度均值
				double meanIn;//缺陷区域灰度均值
				Mat maskGray;//做掩膜后的图像

				int grayValueSum = 0;//像素灰度总和
				int pixelsNum = 0;//像素点个数
				bitwise_and(sidelightSuspect, mask, maskGray);//侧光图做掩膜
				//计算侧光图上缺陷区域灰度均值
				for (int i = 0; i < maskGray.cols; i++)
				{
					for (int j = 0; j < maskGray.rows; j++)
					{
						if (maskGray.at<uchar>(j, i) > 0)
						{
							grayValueSum += maskGray.at<uchar>(j, i);
							pixelsNum++;
						}
					}
				}
				meanIn = grayValueSum / (float)pixelsNum;

				//计算侧光图上缺陷外围灰度均值
				grayValueSum = 0;
				pixelsNum = 0;
				bitwise_and(sidelightSuspect, ~mask, maskGray);
				for (int i = 0; i < maskGray.cols; i++)
				{
					for (int j = 0; j < maskGray.rows; j++)
					{
						if (maskGray.at<uchar>(j, i) <= meanIn && maskGray.at<uchar>(j, i) > 0)
						{
							grayValueSum += maskGray.at<uchar>(j, i);
							pixelsNum++;
						}
					}
				}
				meanOut = grayValueSum / (float)pixelsNum;
				double removeScratch = meanIn - meanOut;//缺陷区域与周围灰度差值(整体性)
				//计算方差
				cv::Mat meanGray;
				cv::Mat stdDev;
				for (int k = 0; k < sidelightSuspect.cols; k++)
				{
					for (int t = 0; t < sidelightSuspect.rows; t++)
					{
						if (sidelightSuspect.at<uchar>(t, k) == 0)
						{
							sidelightSuspect.at<uchar>(t, k) = meanOut;
						}
					}
				}
				cv::meanStdDev(sidelightSuspect, meanGray, stdDev);
				double sideLightStddev = stdDev.at<double>(0, 0);
				Mat white_defect = src_red(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//白底疑似缺陷区域图像
				double meanwhite_in = mean(white_defect, mask)[0];
				double meanwhite_out = mean(white_defect, ~mask)[0];
				double meanwhite_diff = meanwhite_in - meanwhite_out;
				if (removeScratch < 14 && abs(meanwhite_diff) >1.5)//lackscratchth 5 //sideLightStddev >3.2
				{
					result = true;
					CvPoint top_lef4 = cvPoint(x_lt, y_lt);
					CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
					rectangle(src_red, top_lef4, bottom_right4, Scalar(0, 255, 0), 5, 8, 0);
					string information = "th:" + to_string(removeScratch) + " area " + to_string(area) + " width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
					putText(src_red, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
					imwrite("D:\\SX\\" + name + "_lackline_information_bgr.bmp", src_red);
					*mresult = src_red;
					*causecolor = "少线";
					break;


				}
			}
		}
	}

	stack<int> x_stack;
	stack<int> y_stack;
	if (result == false)
	{
		int min_x;
		int max_x;
		for (vector<int>::size_type i = 0; i < X_list.size(); i++)
		{
			int temp_x = X_list.at(i);
			int temp_y = Y_list.at(i);

			if (count_x.at(temp_x) >= 5)//26 >10
			{
				if (!(x_stack.empty()))
				{
					int temp = x_stack.top();
					if (temp != temp_x)
					{
						//x_stack.pop();
						x_stack.push(temp_x);
					}
				}
				else
					x_stack.push(temp_x);
			}
			else
			{
				count_x[temp_x]++;
			}
			if (count_y.at(temp_y) >= 5)//26
			{
				if (!(y_stack.empty()))
				{
					int temp = y_stack.top();
					if (temp != temp_y)
					{
						//y_stack.pop();
						y_stack.push(temp_y);
					}
				}
				else
					y_stack.push(temp_y);
			}
			else
			{
				count_y[temp_y]++;
			}
		}
		Mat temp_mask = Mat::zeros(imageBinary.rows, imageBinary.cols, CV_8UC1);
		while (!(x_stack.empty()) && result == false)
		{
			int index = x_stack.top();
			x_stack.pop();

			min_x = Y_list.at(0);
			max_x = Y_list.at(0);
			for (vector<int>::size_type i = 0; i < X_list.size(); i++)
			{
				int temp_y = Y_list.at(i);
				if (X_list.at(i) == index)
				{
					if (temp_y < min_x)
					{
						min_x = temp_y;
					}
					if (temp_y > max_x)
					{
						max_x = temp_y;
					}
				}
			}
			temp_mask(Rect(min_x, index, max_x - min_x, 1)) = uchar(255);
			int X_1 = min_x;//矩形左上角X坐标值
			int Y_1 = index;//矩形左上角Y坐标值
			int X_2 = max_x;//矩形右下角X坐标值
			int Y_2 = index;//矩形右下角Y坐标值
			int border = 3;//选定框边界宽度
			int x_lt = X_1 - border;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border;
			if (x_rt > src_red.size[1] - 1)
			{
				x_rt = src_red.size[1] - 1;
			}
			int y_rt = Y_2 + border;
			if (y_rt > src_red.size[0] - 1)
			{
				y_rt = src_red.size[0] - 1;
			}
			//侧光图排除明显划痕
			Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
			Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));             //侧光图像疑似贴膜划痕掩膜
			double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
			double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
			double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                        //排除侧光图贴膜划痕的参数
			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
			double removeScratchFlag;
			if (removeScratchArea > 10000)
				removeScratchFlag = 5;
			else
				removeScratchFlag = 3;
			if (abs(removeScratch) < removeScratchFlag)  //lackscratchth
			{
				result = true;
				CvPoint top_lef4 = cvPoint(x_lt, y_lt);
				CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
				rectangle(src_red, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
				string information = "th:" + to_string(removeScratch) + " area " + to_string(removeScratchArea);// +" width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
				putText(src_red, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
				imwrite("D:\\SX\\" + name + "_lackline_bgr.bmp", src_red);
			}
		}
		while (!(y_stack.empty()) && result == false)
		{
			int index = y_stack.top();
			y_stack.pop();

			min_x = X_list.at(0);
			max_x = X_list.at(0);
			for (vector<int>::size_type i = 0; i < X_list.size(); i++)
			{
				int temp_x = X_list.at(i);
				if (Y_list.at(i) == index)
				{

					if (temp_x < min_x)
					{
						min_x = temp_x;
					}
					if (temp_x > max_x)
					{
						max_x = temp_x;
					}
				}
			}
			temp_mask(Rect(index, min_x, 1, max_x - min_x)) = uchar(255);
			int X_1 = index;//矩形左上角X坐标值
			int Y_1 = min_x;//矩形左上角Y坐标值
			int X_2 = index;//矩形右下角X坐标值
			int Y_2 = max_x;//矩形右下角Y坐标值
			int border = 3;//选定框边界宽度
			int x_lt = X_1 - border;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border;
			if (x_rt > src_red.size[1] - 1)
			{
				x_rt = src_red.size[1] - 1;
			}
			int y_rt = Y_2 + border;
			if (y_rt > src_red.size[0] - 1)
			{
				y_rt = src_red.size[0] - 1;
			}
			//侧光图排除明显划痕
			Mat sidelightSuspect = gray_ceguang(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));//侧光图像疑似贴膜划痕图像
			Mat mask = temp_mask(Rect(x_lt + 1, y_lt + 1, x_rt - x_lt - 2, y_rt - y_lt - 2));               //侧光图像疑似贴膜划痕掩膜
			double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
			double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
			double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                        //排除侧光图贴膜划痕的参数
			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
			double removeScratchFlag;
			if (removeScratchArea > 10000)
				removeScratchFlag = 5;
			else
				removeScratchFlag = 3;
			if (abs(removeScratch) < removeScratchFlag)  //lackscratchth
			{
				result = true;
				CvPoint top_lef4 = cvPoint(x_lt, y_lt);
				CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
				rectangle(src_red, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
				string information = "th:" + to_string(removeScratch) + " area " + to_string(removeScratchArea);// +" width:" + to_string(min(w, h)) + " length:" + to_string(max(w, h));
				putText(src_red, information, Point(40, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
				imwrite("D:\\SX\\" + name + "_lackline_bgr.bmp", src_red);
			}
		}
	}


	return result;
}

bool edgelackline(Mat src_red, Mat ceguang, Mat* mresult, string* causecolor, string name)
{

	//幂操作
	bool result = false;
	Mat borderImg;
	Mat front = src_red.clone();
	Mat right = front(Rect(2820, 0, 180, front.rows)).clone();
	Mat rightce = ceguang(Rect(2820, 0, 180, front.rows)).clone();
	Mat shieldMask;
	Mat shieldMask2;
	int bored = 30;
	vector<vector<Point>> contours_mask;
	vector<int>::size_type R = 0;
	Mat shieldMask3 = Mat::zeros(src_red.size(), CV_8UC1);
	threshold(right, shieldMask, 150, 255, CV_THRESH_BINARY);
	threshold(right, shieldMask2, 40, 255, CV_THRESH_BINARY_INV);
	Mat shieldMask4 = shieldMask + shieldMask2;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(shieldMask4, shieldMask4, element2);//扩大缺陷
	shieldMask4 = ~shieldMask4;
	findContours(shieldMask4, contours_mask, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	std::sort(contours_mask.begin(), contours_mask.end(), compareContourAreas);
	shieldMask4 = 0;
	drawContours(shieldMask4, contours_mask, R, 255, FILLED, 8);
	double meanmain = mean(right, shieldMask4)[0];
	Mat th1_test1, th1_test2;
	bitwise_and(right, shieldMask4, th1_test2);
	normalize(th1_test2, th1_test1, 0, 255, CV_MINMAX); //归一化
	convertScaleAbs(th1_test1, th1_test1);              //相当于增强
	medianBlur(th1_test1, th1_test1, 3);
	//right = Contrast(right, 0.2, 0.7*meanmain);

	//边界镜像扩充，自适应分割
	Mat th1_test3 = th1_test1(Rect(0, 0, th1_test1.cols - 15, th1_test1.rows));
	copyMakeBorder(th1_test3, borderImg, 0, 0, 0, 60, BORDER_REFLECT_101);
	//Mat thLeftEdge;
	Mat thRightEdge;
	Mat cethRightEdge;
	medianBlur(rightce, rightce, 3);
	adaptiveThreshold(rightce, cethRightEdge, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -2);//21   -1
	cethRightEdge(Rect(0, 300, cethRightEdge.cols, cethRightEdge.rows - 300)) = uchar(0);
	adaptiveThreshold(borderImg, thRightEdge, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 21, 2);//21   -1

	morphologyEx(cethRightEdge, cethRightEdge, MORPH_OPEN, element);   //开运算形态学操作。可以减少噪点
	dilate(cethRightEdge, cethRightEdge, element2);

	morphologyEx(thRightEdge, thRightEdge, MORPH_OPEN, element);   //开运算形态学操作。可以减少噪点
	dilate(thRightEdge, thRightEdge, element);
	Mat thRightEdge1;
	thRightEdge1 = thRightEdge(Rect(0, 0, 175, thRightEdge.rows));
	thRightEdge1(Rect(0, 0, 15, thRightEdge1.rows)) = uchar(0);
	thRightEdge1(Rect(0, 0, thRightEdge1.cols, 30)) = uchar(0);
	thRightEdge1(Rect(0, thRightEdge1.rows - 30, thRightEdge1.cols, 30)) = uchar(0);
	bitwise_and(thRightEdge1, shieldMask4(Rect(0, 0, 175, thRightEdge.rows)), thRightEdge1);
	bitwise_and(thRightEdge1, ~cethRightEdge(Rect(0, 0, 175, cethRightEdge.rows)), thRightEdge1);

	vector<vector<Point>>contoursEdge;
	findContours(thRightEdge1, contoursEdge, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contoursEdge.begin(), contoursEdge.end(), compareContourAreas);
	vector<Rect> smallBoundRect(contoursEdge.size());
	for (vector<int> ::size_type i = 0; i < contoursEdge.size(); i++) {

		RotatedRect rect = minAreaRect(contoursEdge[i]);  //包覆轮廓的最小斜矩形
		Point p = rect.center;
		double angle = rect.angle;
		double area = contourArea(contoursEdge[i]);
		Moments m = moments(contoursEdge[i]);
		Mat temp_mask = Mat::zeros(thRightEdge1.rows, thRightEdge1.cols, CV_8UC1);
		drawContours(temp_mask, contoursEdge, i, 255, FILLED, 8);
		double smallRatio1 = max(rect.size.width / rect.size.height, rect.size.height / rect.size.width);

		//面积判定
		if (area < 200) {  //905,801,1493,390,450,647,
			break;
		}
		int x_point = int(m.m10 / m.m00);
		int y_point = int(m.m01 / m.m00);
		if (x_point < 0 || x_point >= thRightEdge1.cols || y_point < 0 || y_point >= thRightEdge1.rows)
		{
			continue;
		}
		//正交外接矩形
		smallBoundRect[i] = boundingRect(Mat(contoursEdge[i]));
		int smallX_1 = smallBoundRect[i].tl().x;//矩形左上角X坐标值
		int smallY_1 = smallBoundRect[i].tl().y;//矩形左上角Y坐标值
		int smallX_2 = smallBoundRect[i].br().x;//矩形右下角X坐标值
		int smallY_2 = smallBoundRect[i].br().y;//矩形右下角Y坐标值


		Mat white_defect = right(Rect(smallX_1, smallY_1, smallX_2 - smallX_1 - 1, smallY_2 - smallY_1 - 1));//白底图像疑似反光划痕图像
		Mat mask = temp_mask(Rect(smallX_1, smallY_1, smallX_2 - smallX_1 - 1, smallY_2 - smallY_1 - 1));             //侧光图像疑似贴膜划痕掩膜
		cv::Mat meanGray;
		cv::Mat stdDev;
		cv::meanStdDev(white_defect, meanGray, stdDev);
		double meanwhite_in = mean(white_defect, mask)[0];
		double meanwhite_out = mean(white_defect, ~mask)[0];
		double meanwhite_diff = meanwhite_in - meanwhite_out;
		//长宽比判定
		RotatedRect  smallRotRect = minAreaRect(Mat(contoursEdge[i]));
		float w = smallBoundRect[i].width;
		float h = smallBoundRect[i].height;
		double smallHei = smallRotRect.size.height;
		double smallWid = smallRotRect.size.width;
		double smallRatio = max(smallHei / smallWid, smallWid / smallHei);
		if (smallRatio > 2.0 && w > 30 && h < 16 && (angle<-85 || angle>-5) && meanwhite_diff < 0) {
			*causecolor = "边缘少线";
			result = true;
			CvPoint small_lt = cvPoint(smallX_1, smallY_1);
			CvPoint small_br = cvPoint(smallX_2, smallY_2);
			rectangle(right, small_lt, small_br, Scalar(255, 255, 255), 5, 8, 0);
			right.copyTo(front(Rect(2820, 0, right.cols, right.rows)));
			*mresult = front;
			break;
		}
	}
	return result;


}

/*=========================================================
  * 函 数 名: Stripe
  * 功能描述: 条纹检测
  * 输入：    主相机灰底图像
  =========================================================*/
bool Stripe(Mat gray, Mat* mresult, string* causecolor)
{
	bool result = false;
	Mat Gray_Enh;
	Mat col;
	CvPoint top_lef4;
	CvPoint bottom_right4;

	int num = 0; //条纹出现次数
	int Num_Stripe = 0;
	double m, n, p, o, r;    //二阶差分中间量
	double Diff_Thre = 0.22;    //二阶差分判断阈值  2.5-0.22

	Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
	clahe->apply(gray, Gray_Enh);
	Mat img_gray = Gray_Enh.clone();
	for (int col_line = 100; col_line < img_gray.cols - 220; col_line++)//列进行编列**排除边界处突亮影响
	{
		col = img_gray.colRange(col_line, col_line + 5).clone();
		m = mean(col)[0];

		col = img_gray.colRange(col_line + 5, col_line + 10).clone();
		n = mean(col)[0];

		p = m - n;   //二阶差分减数

		col = img_gray.colRange(col_line + 10, col_line + 15).clone();
		m = mean(col)[0];

		col = img_gray.colRange(col_line + 15, col_line + 20).clone();
		n = mean(col)[0];

		o = m - n;   //二阶差分被减数

		r = o - p;   //改变二阶差分计算方式

		if (r < 0)
			r = 0;
		cout << "r:" << r << endl;
		if (r > Diff_Thre)
		{
			num++;
		}
		if (r < Diff_Thre)
		{
			cout << "num:" << num << endl;
			if (num < 30 && num >5)  //尖峰的宽度特征
			{
				Num_Stripe++;
				top_lef4 = cvPoint(col_line + 8, 0);
				bottom_right4 = cvPoint(col_line + 13, 1775);

			}
			num = 0;
		}
	}

	if (Num_Stripe > 8.1)
	{
		result = true;
		rectangle(gray, top_lef4, bottom_right4, Scalar(255, 255, 255), 1, 8, 0);
		imwrite("D:\\Stripe-information.bmp", gray);
	}
	if (result == true)
	{
		*mresult = gray;
		*causecolor = "条纹";
	}
	return result;
}


/*=========================================================
  * 函 数 名: eliminateBorder
  * 功能描述: 消除边缘的白色区域
  * 输入：
  =========================================================*/
Mat eliminateBorder(Mat thResult, double meanTh, int leftCompensate, int rightCompensate, int topCompensate, int bottomCompensate)//
{
	int terminationPosition;
	//找到二值图左测白色区域的终止位置，并把这一列变成黑色，以割断于边缘白色区域粘在一起的缺陷
	for (int k = 0; k < thResult.cols; k++)
	{
		if (mean(thResult(Rect(k, 0, 1, thResult.rows)))[0] < meanTh)
		{
			terminationPosition = k + leftCompensate;
			break;
		}
	}
	thResult(Rect(0, 0, terminationPosition, thResult.rows)) = uchar(0);
	//找到二值图右测白色区域的起始位置，并把这一列变成黑色，以割断于边缘白色区域粘在一起的缺陷
	for (int k = thResult.cols - 1; k > 0; k--)
	{
		if (mean(thResult(Rect(k, 0, 1, thResult.rows)))[0] < meanTh)
		{
			terminationPosition = k - rightCompensate;
			break;
		}
	}
	thResult(Rect(terminationPosition, 0, thResult.cols - terminationPosition, thResult.rows)) = uchar(0);
	//找到二值图上侧白色区域的终止位置，并把这一列变成黑色，以割断于边缘白色区域粘在一起的缺陷
	for (int k = 0; k < thResult.rows; k++)
	{
		if (mean(thResult(Rect(0, k, thResult.cols, 1)))[0] < meanTh)
		{
			terminationPosition = k + topCompensate;
			break;
		}
	}
	thResult(Rect(0, 0, thResult.cols, terminationPosition)) = uchar(0);
	//找到二值图下侧白色区域的起始位置，并把这一列变成黑色，以割断于边缘白色区域粘在一起的缺陷
	for (int k = thResult.rows - 1; k > 0; k--)
	{
		if (mean(thResult(Rect(0, k, thResult.cols, 1)))[0] < meanTh)
		{
			terminationPosition = k - bottomCompensate;
			break;
		}
	}
	thResult(Rect(0, terminationPosition, thResult.cols, thResult.rows - terminationPosition)) = uchar(0);
	return thResult;
}


/*=========================================================
  * 函 数 名: adaptiveThresholdCustom
  * 功能描述: 自适应二值化
  * 输入：
  =========================================================*/
void adaptiveThresholdCustom(const cv::Mat& src, cv::Mat& dst, double maxValue, int method, int type, int blockSize, double delta, double ratio, bool filledsSrategy)
{
	CV_Assert(src.type() == CV_8UC1);               // 原图必须是单通道无符号8位,CV_Assert（）若括号中的表达式值为false，则返回一个错误信息
	CV_Assert(blockSize % 2 == 1 && blockSize > 1);	// 块大小必须大于1，并且是奇数
	CV_Assert(maxValue > 0);                        //二值图像最大值
	CV_Assert(ratio > DBL_EPSILON);	                //输入均值比例系数
	Size size = src.size();							//源图像的尺寸
	Mat _dst(size, src.type());						//目标图像的尺寸
	Mat mean;	                                    //存放均值图像
	if (src.data != _dst.data)
		mean = _dst;


	int top = (blockSize - 1) * 0.5;     //填充的上边界行数
	int bottom = (blockSize - 1) * 0.5;  //填充的下边界行数
	int left = (blockSize - 1) * 0.5;	   //填充的左边界行数
	int right = (blockSize - 1) * 0.5;   //填充的右边界行数
	int border_type = BORDER_CONSTANT; //边界填充方式
	Mat src_Expand;	                   //对原图像进行边界扩充

	Mat topImage = src(Rect(0, 0, src.cols, 1));//上边界一行图像

	cv::Scalar color = cv::mean(topImage) * 0.5;//35-80之间均可以  该值需要确定

	//Scalar color = Scalar(50);//35-80之间均可以
	if (filledsSrategy == false)
	{
		copyMakeBorder(src, src_Expand, top, bottom, left, right, border_type, color); //函数可以复制图像并制作边界
	}
	else
	{
		copyMakeBorder(src, src_Expand, top, bottom, left, right, BORDER_REFLECT_101);
	}

	if (method == ADAPTIVE_THRESH_MEAN_C)
	{
		/*
		 @param src 单通道灰度图
		 @param dst 单通道处理后的图
		 @param int类型的ddepth，输出图像的深度
		 @param Size类型的ksize，内核的大小
		 @param Point类型的anchor，表示锚点
		 @param bool类型的normaliz,即是否归一化
		 @param borderType 图像外部像素的某种边界模式
		 */
		 //方框滤波，模糊
		boxFilter(src_Expand, mean, src.type(), Size(blockSize, blockSize), Point(-1, -1), true, BORDER_CONSTANT);
	}
	else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
	{
		GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT); //高斯滤波
	}
	else
		CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");

	mean = mean(cv::Rect(top, top, src_Expand.cols - top * 2, src_Expand.rows - top * 2)); //删除扩充的图像边界

	int i, j;
	uchar imaxval = saturate_cast<uchar>(maxValue);	 //防止溢出            //将maxValue由double类型转换为uchar型
	int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);   //将idelta由double类型转换为int型
	if (src.isContinuous() && mean.isContinuous() && _dst.isContinuous())  //返回bool值，判断存储是否连续。
	{
		size.width *= size.height;     //看成一维数组
		size.height = 1;
	}
	for (i = 0; i < size.height; i++)
	{
		const uchar* sdata = src.data + src.step * i;		   //指向源图像//data（uchar*）成员就是指向图像数据的第一个字节
		const uchar* mdata = mean.data + mean.step * i;		   //指向均值图
		uchar* ddata = _dst.data + _dst.step * i;	           //指向输出图//为区分行数
		for (j = 0; j < size.width; j++)
		{
			//每个像素的二值化阈值都不一样
			double Thresh = mdata[j] * ratio - idelta;	        //阈值
			if (CV_THRESH_BINARY == type)	                    //S>T时为imaxval
			{
				ddata[j] = sdata[j] > Thresh ? imaxval : 0;
			}
			else if (CV_THRESH_BINARY_INV == type)	            //S<T时为imaxval
			{
				ddata[j] = sdata[j] > Thresh ? 0 : imaxval;
			}
			else
				CV_Error(CV_StsBadFlag, "Unknown/unsupported threshold type");
		}
	}
	dst = _dst.clone();
}

bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_lightleak, Mat *Mwhite, Mat *Mblack, Mat *Mlightleak, Mat *M_white_abshow)
{
	bool Ext_Result_Main = false;													//显示异常标志位
	Mat src = _src.getMat();                                                        //输入源图像
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像
	if (src.type() == CV_8UC1)														//若输入8位图
	{
		src = src.clone();
		threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化
	}
	else
	{
		/*Mat image[3];
		split(src, image);
		Mat img1 = image[0].clone();
		Mat img2 = image[1].clone();
		Mat img3 = image[2].clone();*/
		Mat imageBinary3 = Mat::zeros(src.size(), CV_8UC1);
		int rowNumber = imageBinary3.rows;  //行数
		int colNumber = imageBinary3.cols;  //列数 x 通道数=每一行元素的个数

		//双重循环，遍历所有的像素值
		//for (int i = 0; i < rowNumber; i++)  //行循环
		//{
		//	uchar* data = imageBinary3.ptr<uchar>(i);  //获取第i行的首地址
		//	for (int j = 0; j < colNumber; j++)   //列循环
		//	{
		//		// ---------【开始处理每个像素】-------------   

		//		int max1 = max(max(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i, j)[1]), src.at<Vec3b>(i, j)[2]);
		//	/*	if (max1 == 255) {
		//			cout << "0" << endl;
		//		}*/
		//	/*	int min1 = min(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i, j)[2])/2;*/
		//		/*data[j] = (src.at<Vec3b>(i, j)[0]+ src.at<Vec3b>(i, j)[1]+ src.at<Vec3b>(i, j)[2])/3;*/
		//         data[j] =max1;
		//		// ----------【处理结束】---------------------
		//	}  //行处理结束
		//}
		//threshold(imageBinary3, binaryImage, 35, 255, CV_THRESH_BINARY);							//二值化
		cvtColor(src, src, CV_BGR2GRAY);										    //灰度化彩色图
		//threshold(src, binaryImage, 35, 255, CV_THRESH_BINARY);							//二值化
		//src =img1;
		threshold(src, binaryImage, 20, 255, CV_THRESH_BINARY);							//二值化
	}
	CV_Assert(src.depth() == CV_8U);                                                //8位无符号

	 // 主相机ROI提取 -> 2022.5.24 刘老屯
	// 遍历像素，将背景涂黑
	for (int m = 0; m < src.rows; m++) {
		uchar* t = src.ptr<uchar>(m);
		for (int n = 0; n < src.cols; n++) {
			if (t[n] > 180) {
				(t[n]) = 0;
			}
		}
	}
	threshold(src, binaryImage, 35, 255, CV_THRESH_BINARY);
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7)); //闭操作结构元素
	morphologyEx(binaryImage, binaryImage, CV_MOP_OPEN, element);   //闭运算形态学操作。可以减少噪点
	Mat bin = binaryImage.clone();
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4), src_corner_lightleak(4), src_corner_abshow(4);   //四个边相交得到角点坐标，漏光角点，显示异常角点
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			//正接矩阵坐标点信息
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 3000000 && area < 5300000)   //小屏幕的面积阈值会缩小
		{
			displayError_Areasignal++;
			rect = boundingRect(contours[i]);
			x1 = rect.tl().x;//左上角
			y1 = rect.tl().y;//左上角
			x2 = rect.tl().x;//左下角
			y2 = rect.br().y;//右下角
			x3 = rect.br().x;//右下角
			y3 = rect.br().y;//右下角
			x4 = rect.br().x;//右上角
			y4 = rect.tl().y;//右上角
			int radianEliminate = 150;
			int deviation = 150; //1.77
//            int radianEliminate = 200;
//            int deviation = 200;
			for (int j = 0; j < contours[i].size(); j++)
			{
				//左侧点集
				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
					leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//右侧点集
				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1)*0.3 &&abs(contours[i][j].x - x3) < deviation ||
					contours[i][j].y > y1 + (y2 - y1)*0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
					rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//上侧点集
				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y1) < deviation)
					upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//下侧点集
				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y2) < deviation)
					downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
			}
		}
	}
	if (leftLinePoint.size() == 0 || rightLinePoint.size() == 0 || upLinePoint.size() == 0 || downLinePoint.size() == 0)
		displayError_Areasignal = 0;
	//根据轮廓面积判定显示异常
	if (displayError_Areasignal > 0)
		Ext_Result_Main = false;
	else
		Ext_Result_Main = true;
	//未提取到屏幕判定显示异常提取边缘角落
	if (displayError_Areasignal == 0)
	{
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);
		*Mlightleak = cv::getPerspectiveTransform(src_points, dst_points);
		*Mblack = cv::getPerspectiveTransform(src_points, dst_points);
		*M_white_abshow = cv::getPerspectiveTransform(src_points, dst_points);
	}
	//正常屏幕提取屏幕的四个角点
	else
	{
		fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
		fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
		fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
		fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线

		src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
		src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
		src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
		src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点
		Mat edge_img = bin.clone();

		line(edge_img, src_corner[0], src_corner[1], (0, 0, 255), 3, cv::LINE_AA);//绘制直线
		line(edge_img, src_corner[0], src_corner[3], (0, 0, 255), 3, cv::LINE_AA);//绘制直线
		line(edge_img, src_corner[2], src_corner[3], (0, 0, 255), 3, cv::LINE_AA);//绘制直线
		line(edge_img, src_corner[2], src_corner[1], (0, 0, 255), 3, cv::LINE_AA);//绘制直线
																	//对4个角点的坐标位置进行微调（白底图以及黑底图）
		src_corner[0].x = src_corner[0].x - border_white;
		src_corner[0].y = src_corner[0].y - border_white;
		src_corner[1].x = src_corner[1].x - border_white;
		src_corner[1].y = src_corner[1].y + border_white;
		src_corner[2].x = src_corner[2].x + border_white;
		src_corner[2].y = src_corner[2].y + border_white;
		src_corner[3].x = src_corner[3].x + border_white;
		src_corner[3].y = src_corner[3].y - border_white;
		int count = 0;
		//while (bin.at<uchar>(src_corner[0]) != 255)
		//{
		//	count++;
		//	//Mat src_r = bin.clone();
		//	//circle(src_r, src_corner[0],2, Scalar(255, 255, 255));
		//	if (bin.at<uchar>(cvRound(src_corner[0].y + 1), cvRound(src_corner[0].x + 1)) == 255)
		//	{
		//		break;
		//	}
		//	else
		//	{
		//		if (bin.at<uchar>(cvRound(src_corner[0].y + 3), cvRound(src_corner[0].x)) == 255 || bin.at<uchar>(cvRound(src_corner[0].y + 20), cvRound(src_corner[0].x)) == 255)
		//		{
		//			src_corner[0].y = src_corner[0].y + 1;
		//		}
		//		else if (bin.at<uchar>(cvRound(src_corner[0].y), cvRound(src_corner[0].x + 3)) == 255 || bin.at<uchar>(cvRound(src_corner[0].y), cvRound(src_corner[0].x + 20)) == 255)
		//		{
		//			src_corner[0].x = src_corner[0].x + 1;
		//		}
		//		else if (bin.at<uchar>(cvRound(src_corner[0].y + 1), cvRound(src_corner[0].x + 1)) != 255)
		//		{
		//			src_corner[0].x = src_corner[0].x + 1;
		//			src_corner[0].y = src_corner[0].y + 1;
		//		}
		//		else
		//		{
		//			break;
		//		}
		//	}
		//	if (src_corner[0].x >= bin.cols / 2 || src_corner[0].y >= bin.rows / 2)
		//	{
		//		break;
		//	}
		//	if (src_corner[0].x <= 1 || src_corner[0].x >= bin.cols - 1 || src_corner[0].y < 1 || src_corner[0].y > bin.rows - 1)
		//	{
		//		break;
		//	}
		//	if (count == 200)//太多次直接退出
		//	{
		//		count = 0;
		//		break;
		//	}
		//}
		//count = 0;
		//while (bin.at<uchar>(src_corner[1]) != 255)
		//{
		//	count++;
		//	//Mat src_r = bin.clone();
		//	//circle(src_r, src_corner[1], 2, Scalar(255, 255, 255));
		//	if (bin.at<uchar>(cvRound(src_corner[1].y - 1), cvRound(src_corner[1].x + 1)) == 255)
		//	{
		//		break;
		//	}
		//	else
		//	{
		//		if (bin.at<uchar>(cvRound(src_corner[1].y - 3), cvRound(src_corner[1].x)) == 255 || bin.at<uchar>(cvRound(src_corner[1].y - 20), cvRound(src_corner[1].x)) == 255)
		//		{
		//			src_corner[1].y = src_corner[1].y - 1;
		//		}
		//		else if (bin.at<uchar>(cvRound(src_corner[1].y), cvRound(src_corner[1].x + 3)) == 255 || bin.at<uchar>(cvRound(src_corner[1].y), cvRound(src_corner[1].x + 20)) == 255)
		//		{
		//			src_corner[1].x = src_corner[1].x + 1;
		//		}
		//		else if (bin.at<uchar>(cvRound(src_corner[1].y - 1), cvRound(src_corner[1].x + 1)) != 255)
		//		{
		//			src_corner[1].x = src_corner[1].x + 1;
		//			src_corner[1].y = src_corner[1].y - 1;
		//		}
		//		else
		//		{
		//			break;
		//		}
		//	}
		//	if (src_corner[1].x >= bin.cols / 2 || src_corner[1].y <= bin.rows / 2)
		//	{
		//		break;
		//	}
		//	if (src_corner[1].x <= 1 || src_corner[1].x >= bin.cols - 1 || src_corner[1].y < 1 || src_corner[1].y > bin.rows - 1)
		//	{
		//		break;
		//	}
		//	if (count == 200)
		//	{
		//		break;
		//	}
		//}
		//count = 0;
		//while (bin.at<uchar>(src_corner[2]) != 255)
		//{
		//	count++;
		//	//Mat src_r = bin.clone();
		//	//circle(src_r, src_corner[2], 8, Scalar(255, 255, 255));
		//	if (bin.at<uchar>(cvRound(src_corner[2].y - 1), cvRound(src_corner[2].x - 1)) == 255)
		//	{
		//		break;
		//	}
		//	else
		//	{
		//		if (bin.at<uchar>(cvRound(src_corner[2].y - 3), cvRound(src_corner[2].x)) == 255 || bin.at<uchar>(cvRound(src_corner[2].y - 20), cvRound(src_corner[2].x)) == 255)
		//		{
		//			src_corner[2].y = src_corner[2].y - 1;
		//		}
		//		else if (bin.at<uchar>(cvRound(src_corner[2].y), cvRound(src_corner[2].x - 3)) == 255 || bin.at<uchar>(cvRound(src_corner[2].y), cvRound(src_corner[2].x - 20)) == 255)
		//		{
		//			src_corner[2].x = src_corner[2].x - 1;
		//		}
		//		else if (bin.at<uchar>(cvRound(src_corner[2].y - 1), cvRound(src_corner[2].x - 1)) != 255)
		//		{
		//			src_corner[2].x = src_corner[2].x - 1;
		//			src_corner[2].y = src_corner[2].y - 1;
		//		}
		//		else
		//		{
		//			break;
		//		}
		//	}
		//	//if (src_corner[2].x <= bin.cols / 2 || src_corner[2].y <= bin.rows / 2)
		//	//{
		//	//	break;
		//	//}
		//	if (src_corner[2].x <= 1 || src_corner[2].x >= bin.cols - 1 || src_corner[2].y < 1 || src_corner[2].y > bin.rows - 1)
		//	{
		//		break;
		//	}
		//	if (count == 200)
		//	{
		//		count = 0;
		//		break;
		//	}
		//}
		//count = 0;
		//while (bin.at<uchar>(src_corner[3]) != 255)
		//{
		//	count++;
		//	//Mat src_r = bin.clone();
		//	//circle(src_r, src_corner[1], 2, Scalar(255, 255, 255));
		//	if (bin.at<uchar>(cvRound(src_corner[3].y + 1), cvRound(src_corner[3].x - 1)) == 255)
		//	{
		//		break;
		//	}
		//	else
		//	{
		//		if (bin.at<uchar>(cvRound(src_corner[3].y + 3), cvRound(src_corner[3].x)) == 255 || bin.at<uchar>(cvRound(src_corner[3].y + 20), cvRound(src_corner[3].x)) == 255)
		//		{
		//			src_corner[3].y = src_corner[3].y + 1;
		//		}
		//		else if (bin.at<uchar>(cvRound(src_corner[3].y), cvRound(src_corner[3].x - 3)) == 255 || bin.at<uchar>(cvRound(src_corner[3].y), cvRound(src_corner[3].x - 20)) == 255)
		//		{
		//			src_corner[3].x = src_corner[3].x - 1;
		//		}
		//		else if (bin.at<uchar>(cvRound(src_corner[3].y + 1), cvRound(src_corner[3].x - 1)) != 255)
		//		{
		//			src_corner[3].x = src_corner[3].x - 1;
		//			src_corner[3].y = src_corner[3].y + 1;
		//		}
		//		else
		//		{
		//			break;
		//		}
		//	}
		//	//if (src_corner[3].x <= bin.cols / 2 || src_corner[3].y >= bin.rows / 2)
		//	//{
		//	//	break;
		//	//}
		//	if (src_corner[3].x <= 1 || src_corner[3].x >= bin.cols - 1 || src_corner[3].y < 1 || src_corner[3].y > bin.rows - 1)
		//	{
		//		break;
		//	}
		//	if (count == 200)
		//	{
		//		count = 0;
		//		break;
		//	}
		//}
		//对4个角点的坐标位置进行微调（漏光检测图）
		src_corner_lightleak[0].x = src_corner[0].x - border_lightleak;
		src_corner_lightleak[0].y = src_corner[0].y - border_lightleak;
		src_corner_lightleak[1].x = src_corner[1].x - border_lightleak;
		src_corner_lightleak[1].y = src_corner[1].y + border_lightleak;
		src_corner_lightleak[2].x = src_corner[2].x + border_lightleak;
		src_corner_lightleak[2].y = src_corner[2].y + border_lightleak;
		src_corner_lightleak[3].x = src_corner[3].x + border_lightleak;
		src_corner_lightleak[3].y = src_corner[3].y - border_lightleak;
		//显示异常(白底图)
		src_corner_abshow[0].x = src_corner[0].x - border_white + 10;
		src_corner_abshow[0].y = src_corner[0].y - border_white + 10;
		src_corner_abshow[1].x = src_corner[1].x - border_white + 10;
		src_corner_abshow[1].y = src_corner[1].y + border_white - 10;
		src_corner_abshow[2].x = src_corner[2].x + border_white - 10;
		src_corner_abshow[2].y = src_corner[2].y + border_white - 10;
		src_corner_abshow[3].x = src_corner[3].x + border_white - 10;
		src_corner_abshow[3].y = src_corner[3].y - border_white + 10;
		//line(bin, src_corner[0], src_corner[1], cv::Scalar(0, 0, 0), 2);
		//line(bin, src_corner[1], src_corner[2], cv::Scalar(0, 0, 0), 2);
		//line(bin, src_corner[2], src_corner[3], cv::Scalar(0, 0, 0), 2);
		//line(bin, src_corner[3], src_corner[0], cv::Scalar(0, 0, 0), 2);
		circle(bin, src_corner[0], 8, Scalar(255, 255, 255));
		circle(bin, src_corner[1], 8, Scalar(255, 255, 255));
		circle(bin, src_corner[2], 8, Scalar(255, 255, 255));
		circle(bin, src_corner[3], 8, Scalar(255, 255, 255));
		vector<Point2f> dst_corner(4);
		dst_corner = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
		*Mlightleak = cv::getPerspectiveTransform(src_corner_lightleak, dst_corner);
		*Mblack = cv::getPerspectiveTransform(src_corner, dst_corner);
		*M_white_abshow = cv::getPerspectiveTransform(src_corner_abshow, dst_corner);
	}

	return Ext_Result_Main;
}

void convexSetPretreatment(Mat& src)//引用的方式
{
	//空洞预处理 针对于主相机
	Mat src_copy = src.clone();
	Mat threshold_output;
	vector<vector<Point> > preContours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	//threshold(src, threshold_output, 30, 255, THRESH_BINARY);

	//threshold(src_copy, threshold_output, 30, 255, THRESH_BINARY);
	double meanGray = mean(src_copy)[0];
	threshold(src_copy, threshold_output, meanGray * 0.6, 255, THRESH_BINARY);

	/// Find contours
	findContours(threshold_output, preContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(preContours.size());

	for (size_t i = 0; i < preContours.size(); i++)
	{
		//hull,输出的一个凸包上的点
		convexHull(Mat(preContours[i]), hull[i], false);//凸包对于图像来说其实就是图像轮廓突出点的连线。对于点集合来说就是点集合最外围点的连线，通过这些连线形成的多边形可以刚好将所有的点包裹起来。
	}

	/// Draw contours + hull results
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC1);
	for (size_t i = 0; i < preContours.size(); i++)
	{
		double area = contourArea(preContours[i]);
		if (area > 10000)
		{
			drawContours(drawing, preContours, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
			drawContours(drawing, hull, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
		}
	}
	src = drawing;
}

bool mainCamPersTransMatCalwhite(InputArray _src, int border_white, Mat* Mwhite)
{

	bool isArea_1, isArea_2;														//显示异常标志位
	Mat src = _src.getMat();  //将OutputArray数据转换成Mat类型                      //输入源图像

	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //灰度化彩色图
	//convexSetPretreatment(dst);

	Mat binaryImage;

	threshold(src, binaryImage, 200, 255, CV_THRESH_BINARY_INV);    //反向二值化
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));    //闭操作结构元素，内核
	morphologyEx(binaryImage, binaryImage, CV_MOP_OPEN, element);   //闭运算形态学操作。可以减少噪点

	CV_Assert(src.depth() == CV_8U);    //若()中的结果为false，则返回错误信息       //8位无符号
	Mat bin = binaryImage.clone();
	medianBlur(binaryImage, binaryImage, 5);	//平滑、模糊   					    //中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	sort(contours.begin(), contours.end(), compareContourAreas);
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4), src_corner_lightleak(4), src_corner_abshow(4);   //四个边相交得到角点坐标，漏光角点，显示异常角点
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			//正接矩阵坐标点信息
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		Mat tempMask = Mat::zeros(binaryImage.size(), CV_8UC1);
		drawContours(tempMask, contours, i, 255, -1, 8);
		convexSetPretreatment(tempMask);
		double area = contourArea(contours[i]);
		if (area > 3000000 && area < 5300000)   //小屏幕的面积阈值会缩小
		{
			displayError_Areasignal++;
			rect = boundingRect(contours[i]);
			RotatedRect rect1 = minAreaRect(contours[i]);  //包覆轮廓的最小斜矩形
		/*	cout << "长：" << max(rect1.size.width, rect1.size.height) << endl;
			cout << "宽：" << min(rect1.size.width, rect1.size.height) << endl;
			cout << "角度：" << rect1.angle << endl;*/
			x1 = rect.tl().x;//左上角
			y1 = rect.tl().y;//左上角
			x2 = rect.tl().x;//左下角
			y2 = rect.br().y;//右下角
			x3 = rect.br().x;//右下角
			y3 = rect.br().y;//右下角
			x4 = rect.br().x;//右上角
			y4 = rect.tl().y;//右上角
			int radianEliminate = 150;
			int deviation = 150; //1.77
			for (int j = 0; j < contours[i].size(); j++)
			{
				//左侧点集
				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
					leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//右侧点集
				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1) * 0.3 && abs(contours[i][j].x - x3) < deviation ||
					contours[i][j].y > y1 + (y2 - y1) * 0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
					rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//上侧点集
				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y1) < deviation)
					upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
				//下侧点集
				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y2) < deviation)
					downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
			}
		}
	}

	if (leftLinePoint.size() == 0 || rightLinePoint.size() == 0 || upLinePoint.size() == 0 || downLinePoint.size() == 0)
		displayError_Areasignal = 0;
	//根据轮廓面积判定显示异常
	if (displayError_Areasignal > 0)
		isArea_1 = false;
	if (displayError_Areasignal == 0)
		isArea_1 = true;

	//未提取到屏幕判定显示异常提取边缘角落
	if (displayError_Areasignal == 0)
	{
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);
	}
	//正常屏幕提取屏幕的四个角点
	else
	{
		fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
		fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
		fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
		fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线

		src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
		src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
		src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
		src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点

		//对4个角点的坐标位置进行微调（白底图以及黑底图）
		src_corner[0].x = src_corner[0].x + 8;
		src_corner[0].y = src_corner[0].y + 5;
		src_corner[1].x = src_corner[1].x + 8;
		src_corner[1].y = src_corner[1].y - 5;
		src_corner[2].x = src_corner[2].x - 5;
		src_corner[2].y = src_corner[2].y - 5;
		src_corner[3].x = src_corner[3].x - 5;
		src_corner[3].y = src_corner[3].y + 5;

		vector<Point2f> dst_corner(4);
		dst_corner[0] = Point(0, 0);
		dst_corner[1] = Point(0, 1500);
		dst_corner[2] = Point(3000, 1500);
		dst_corner[3] = Point(3000, 0);

		*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
	}

	return isArea_1;
}


/*=========================================================
* 函 数 名: getPointSlopeCrossPoint
* 功能描述: 求两直线交点
=========================================================*/
Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB)
{
	const double PI = 3.1415926535897;
	Point2f crossPoint;
	double kA = LineA[1] / LineA[0];
	double kB = LineB[1] / LineB[0];
	double theta = atan2(LineB[1], LineB[0]);//反正切函数，返会弧度  -pi到pi
	if (theta == PI * 0.5)
	{
		crossPoint.x = LineB[0];
		crossPoint.y = kA * LineB[0] + LineA[3] - kA * LineA[2];
		return crossPoint;
	}
	double bA = LineA[3] - kA * LineA[2];//x为0时的，y
	double bB = LineB[3] - kB * LineB[2];
	crossPoint.x = (bB - bA) / (kA - kB);
	crossPoint.y = (kA*bB - kB * bA) / (kA - kB);
	return crossPoint;
}

/*=========================================================
* 函 数 名: Gabor7
* 功能描述: gabor滤波
=========================================================*/
Mat Gabor7(Mat img_1)
{                               //(核       ，𝜎    ，𝜃，     𝜆，   𝛾  𝜑，数据类型 )
	Mat kernel1 = getGaborKernel(Size(7, 7), 2.7, CV_PI / 2, 1.0, 1.0, 0, CV_32F);//求卷积核
	float sum = 0.0;
	for (int i = 0; i < kernel1.rows; i++)
	{
		for (int j = 0; j < kernel1.cols; j++)
		{
			sum = sum + kernel1.ptr<float>(i)[j];
		}
	}
	Mat mmm = kernel1 / sum;
	Mat kernel2 = getGaborKernel(Size(7, 7), 2.7, 0, 1.0, 1.0, 0, CV_32F);
	float sum2 = 0.0;
	for (int i = 0; i < kernel2.rows; i++)
	{
		for (int j = 0; j < kernel2.cols; j++)
		{
			sum2 = sum2 + kernel2.ptr<float>(i)[j];
		}
	}
	Mat mmm2 = kernel2 / sum2;
	Mat img_4, img_5;
	filter2D(img_1, img_4, CV_8UC3, mmm);//卷积运算
	filter2D(img_4, img_5, CV_8UC3, mmm2);
	return img_5;
}
/*=========================================================
* 函 数 名: Gabor6
* 功能描述: gabor滤波
=========================================================*/
Mat Gabor6(Mat img_1)
{                               //(核       ，𝜎    ，𝜃，     𝜆，   𝛾  𝜑，数据类型 )
	Mat kernel1 = getGaborKernel(Size(5, 5), 5, CV_PI / 2, 1.0, 0.2, 0, CV_32F);//求卷积核
	float sum = 0.0;
	for (int i = 0; i < kernel1.rows; i++)
	{
		for (int j = 0; j < kernel1.cols; j++)
		{
			sum = sum + kernel1.ptr<float>(i)[j];
		}
	}
	Mat mmm = kernel1 / sum;
	Mat kernel2 = getGaborKernel(Size(5, 5), 5, 0, 1.0, 0.2, 0, CV_32F);
	float sum2 = 0.0;
	for (int i = 0; i < kernel2.rows; i++)
	{
		for (int j = 0; j < kernel2.cols; j++)
		{
			sum2 = sum2 + kernel2.ptr<float>(i)[j];
		}
	}
	Mat mmm2 = kernel2 / sum2;
	Mat img_4, img_5;
	filter2D(img_1, img_4, CV_8UC3, mmm);//卷积运算
	filter2D(img_4, img_5, CV_8UC3, mmm2);
	return img_5;
}
/*=========================================================
* 函 数 名: Gabor9
* 功能描述: 暂时不需要
=========================================================*/
Mat Gabor9(Mat img_1)
{
	int delta = 2.7;
	int kernal_size = 5;
	Mat kernel1 = getGaborKernel(Size(kernal_size, kernal_size), delta, CV_PI / 2, 1.0, 1.0, 0, CV_32F);//求卷积核

	//Mat kernel1 = getGaborKernel(Size(5, 5), 0.5, CV_PI / 2, 3, 0.5, 0, CV_32F);//Size(5, 5),0.5, CV_PI / 2, 3.0, 0.5, 0, CV_32F

	float sum = 0.0;
	for (int i = 0; i < kernel1.rows; i++)
	{
		for (int j = 0; j < kernel1.cols; j++)
		{
			sum = sum + kernel1.ptr<float>(i)[j];
		}
	}
	Mat mmm = kernel1 / sum;//归一化
	Mat kernel2 = getGaborKernel(Size(kernal_size, kernal_size), delta, 0, 1.0, 1.0, 0, CV_32F);
	//Mat kernel2 = getGaborKernel(Size(5, 5), 0.5, 0,  3,  0.5, 0, CV_32F);//垂直卷积核
	float sum2 = 0.0;
	for (int i = 0; i < kernel2.rows; i++)
	{
		for (int j = 0; j < kernel2.cols; j++)
		{
			sum2 = sum2 + kernel2.ptr<float>(i)[j];
		}
	}
	Mat mmm2 = kernel2 / sum2;//归一化
	Mat img_4, img_5;
	filter2D(img_1, img_4, CV_8UC3, mmm);//卷积运算
	filter2D(img_4, img_5, CV_8UC3, mmm2);
	return img_5;
}
/*=========================================================
* 函 数 名: Gabor5
* 功能描述: gabor滤波
=========================================================*/
Mat Gabor5(Mat img_1)
{
	Mat kernel1 = getGaborKernel(Size(5, 5), 1.1, CV_PI / 2, 1.0, 1.0, 0, CV_32F);//求卷积核
	float sum = 0.0;
	for (int i = 0; i < kernel1.rows; i++)
	{
		for (int j = 0; j < kernel1.cols; j++)
		{
			sum = sum + kernel1.ptr<float>(i)[j];
		}
	}
	Mat mmm = kernel1 / sum;
	Mat kernel2 = getGaborKernel(Size(5, 5), 1.1, 0, 1.0, 1.0, 0, CV_32F);
	float sum2 = 0.0;
	for (int i = 0; i < kernel2.rows; i++)
	{
		for (int j = 0; j < kernel2.cols; j++)
		{
			sum2 = sum2 + kernel2.ptr<float>(i)[j];
		}
	}
	Mat mmm2 = kernel2 / sum2;
	Mat img_4, img_5;
	filter2D(img_1, img_4, CV_8UC3, mmm);//卷积运算
	filter2D(img_4, img_5, CV_8UC3, mmm2);
	return img_5;
}

/*=========================================================
* 函 数 名: toushi_white
* 功能描述: 透视变换图像矫正
=========================================================*/
Mat toushi_white(Mat image, Mat M, int border, int length, int width)
{
	Mat perspective;
	cv::warpPerspective(image, perspective, M, cv::Size(length, width), cv::INTER_LINEAR);
	return perspective;
}

void RemoveSmallRegion(Mat &Src, Mat &Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;
	//新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查 
	//初始化的图像全部为0，未检查
	Mat PointLabel = Mat::zeros(Src.size(), CV_8UC1);
	if (CheckMode == 1)//去除小连通区域的白色点
	{
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) < 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//将背景黑色点标记为合格，像素为3
				}
			}
		}
	}
	else//去除孔洞，黑色点像素
	{
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) > 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//如果原图是白色区域，标记为合格，像素为3
				}
			}
		}
	}


	vector<Point2i>NeihborPos;//将邻域压进容器
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			if (PointLabel.at<uchar>(i, j) == 0)//标签图像像素点为0，表示还未检查的不合格点
			{   //开始检查
				vector<Point2i>GrowBuffer;//记录检查像素点的个数
				GrowBuffer.push_back(Point2i(j, i));
				PointLabel.at<uchar>(i, j) = 1;//标记为正在检查
				int CheckResult = 0;


				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					for (int q = 0; q < NeihborCount; q++)
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX < Src.cols&&CurrY >= 0 && CurrY < Src.rows)  //防止越界  
						{
							if (PointLabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer  
								PointLabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查  
							}
						}
					}
				}
				if (GrowBuffer.size() > AreaLimit) //判断结果（是否超出限定的大小），1为未超出，2为超出  
					CheckResult = 2;
				else
				{
					CheckResult = 1;
					RemoveCount++;//记录有多少区域被去除
				}


				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					PointLabel.at<uchar>(CurrY, CurrX) += CheckResult;//标记不合格的像素点，像素值为2
				}
				//********结束该点处的检查**********  


			}
		}


	}


	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域  
	for (int i = 0; i < Src.rows; ++i)
	{
		for (int j = 0; j < Src.cols; ++j)
		{
			if (PointLabel.at<uchar>(i, j) == 2)
			{
				Dst.at<uchar>(i, j) = CheckMode;
			}
			else if (PointLabel.at<uchar>(i, j) == 3)
			{
				Dst.at<uchar>(i, j) = Src.at<uchar>(i, j);

			}
		}
	}
}

Mat gamma(Mat src, double g)
{
	Mat temp;
	src.convertTo(temp, CV_32FC3, 1 / 255.0);
	cv::Mat temp1;
	cv::pow(temp, g, temp1);  //幂操作
	Mat dst;
	temp1.convertTo(dst, CV_8UC1, 255);
	return dst;
}










