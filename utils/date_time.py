#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/15 10:03
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : date_time.py
@Description: 

"""
import datetime


def get_date_time(date_str):
    date_ = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return date_


def get_hour_diff(datetime1, datetime2):
    td = datetime2 - datetime1
    return td.days * 24 + td.seconds / 3600

def get_currentday():
    today = datetime.date.today()
    return today

def get_now_datetime():
    return datetime.datetime.now()

def get_current_yesterday():
    today = get_currentday()
    oneday = datetime.timedelta(days=1)
    yesterday = today - oneday
    return yesterday.strftime("%Y-%m-%d")


def get_yesterday(date_str):
    date_ = get_date_time(date_str)
    yesterday = date_ + datetime.timedelta(days=-1)
    return yesterday


def get_lastyesterday(date_str):
    date_ = get_date_time(date_str)
    lastyesterday = date_ + datetime.timedelta(days=-2)
    return lastyesterday


def get_lastweek_date(date_str):
    date_ = get_date_time(date_str)
    lastweek_date = date_ + datetime.timedelta(days=-7)
    return lastweek_date


def get_lastlastweek_date(date_str):
    date_ = get_date_time(date_str)
    lastlastweek_date = date_ + datetime.timedelta(days=-14)
    return lastlastweek_date


def get_lastweek_start_end_date(date_str):
    date_ = get_date_time(date_str)
    week = date_.strftime("%w")
    week = int(week)
    start_lastweek_date = date_ + datetime.timedelta(days=-(week + 6))
    end_lastweek_date = date_ + datetime.timedelta(days=-week)

    return start_lastweek_date, end_lastweek_date


def add_day(d_date, n):
    return (d_date + datetime.timedelta(days=n)).strftime("%Y-%m-%d")


def get_lastmonth_start_end_date(date_str):
    date_ = get_date_time(date_str)
    start_lastmonth_date = datetime.date(date_.year, date_.month - 1, 1)
    end_lastmonth_date = datetime.date(date_.year, date_.month, 1) - datetime.timedelta(1)
    return start_lastmonth_date, end_lastmonth_date


def get_time_period_frame(start_date, end_date):
    return 0


if __name__ == "__main__":
    startDate = '2020-04-10'
    #
    # date_ = datetime.datetime.strptime(startDate, '%Y-%m-%d')
    # week = date_.strftime("%w")
    # yestoday = date_ + datetime.timedelta(days=-1)
    #
    # week = int(week)
    #
    # start_lastweekday = date_ + datetime.timedelta(days=-(week+6))
    # end_lastweekday = date_ + datetime.timedelta(days=-week)
    #
    # print(yestoday.strftime('%Y-%m-%d'))
    # print(start_lastweekday.strftime('%Y-%m-%d'))
    # print(end_lastweekday.strftime('%Y-%m-%d'))
    # # print((datetime.datetime.today() - datetime.timedelta(days=time.localtime().tm_wday + 1)).strftime("%Y-%m-%d"))
    #
    #
    # first = datetime.date(date_.year, date_.month-1, 1)
    # last = datetime.date(date_.year, date_.month, 1)-datetime.timedelta(1)
    # print(first)
    # print(last)
    # print(type(get_yesterday(startDate)))
    # print(get_lastweek_date(startDate))
    # print(get_lastweek_start_end_date(startDate))
    # print(get_lastmonth_start_end_date(startDate))
    print(datetime.datetime.now().strftime("%H"))