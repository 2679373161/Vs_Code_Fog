#ifndef DBHELPER_H
#define DBHELPER_H
#define _CRT_SECURE_NO_WARNINGS
//#include "stdafx.h"  
#include "sqlite3.h"  
#include <iostream>  
#include <string> 
#include <Windows.h>
using namespace std;


#endif











static bool connect_mysql()
{
	db = QSqlDatabase::addDatabase("QSQLITE");
	//  db.setHostName("D:\workspace\QtWorkSpace\new\new\BM.db");      //�������ݿ���������������Ҫע�⣨�����Ϊ��127.0.0.1�������ֲ������ӣ����Ϊlocalhost)
	//  db.setPort(3306);                                              //�������ݿ�˿ںţ�������һ��
	//db.setDatabaseName(QApplication::applicationDirPath()+"/BM(3)(1)(1).db");  //�����Ŀ¼��û�и��ļ�,����ڱ�Ŀ¼������,�������Ӹ��ļ�

	//qin
	db.setDatabaseName("D:\\BM.db");  //�������ݿ�����������һ��     //�������ݿ�����������һ��
	//db.setDatabaseName("F:\\UI\\BM_0714.db");
	db.setUserName("banma");          //���ݿ��û�����������һ��
	db.setPassword("123456");         //���ݿ����룬������һ��
	db.open();
	if (!db.open())
	{
		mysql = "��������";
		QMessageBox::warning(0, QObject::tr("Database Error"), db.lastError().text());
		qDebug() << "��������" << "connect to mysql error" << db.lastError().text();
		return false;

	}
	else
	{
		mysql = "001-001";
		qDebug() << "connect to mysql successfully";
		qDebug() << qApp->applicationDirPath() << endl;
	}
	return true;

}
static bool close_mysql()
{
	db.close();
	return true;
}

#endif // DBHELPER_H#pragma once
