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
	//  db.setHostName("D:\workspace\QtWorkSpace\new\new\BM.db");      //连接数据库主机名，这里需要注意（若填的为”127.0.0.1“，出现不能连接，则改为localhost)
	//  db.setPort(3306);                                              //连接数据库端口号，与设置一致
	//db.setDatabaseName(QApplication::applicationDirPath()+"/BM(3)(1)(1).db");  //如果本目录下没有该文件,则会在本目录下生成,否则连接该文件

	//qin
	db.setDatabaseName("D:\\BM.db");  //连接数据库名，与设置一致     //连接数据库名，与设置一致
	//db.setDatabaseName("F:\\UI\\BM_0714.db");
	db.setUserName("banma");          //数据库用户名，与设置一致
	db.setPassword("123456");         //数据库密码，与设置一致
	db.open();
	if (!db.open())
	{
		mysql = "不能连接";
		QMessageBox::warning(0, QObject::tr("Database Error"), db.lastError().text());
		qDebug() << "不能连接" << "connect to mysql error" << db.lastError().text();
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
