'''
Date: 2021-06-07 10:49:16
LastEditors: Liuliang
LastEditTime: 2021-06-07 10:56:21
Description: argpaser
'''
import argparse

#######################################################################################
##   注释：
##   1.创建一个解析器 description='Process some integers.'
#######################################################################################
parser = argparse.ArgumentParser(description='Process some integers.')



#######################################################################################
##   注释：
##   2.添加参数.给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的
##     通常，这些调用指定 ArgumentParser 如何获取命令行字符串并将其转换为对象。
##     这些信息在 parse_args() 调用时被存储和使用。
#######################################################################################
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

#######################################################################################
##   注释：
##   3.ArgumentParser 通过 parse_args() 方法解析参数。它将检查命令行，把每个参数转换为适当的
# 类型然后调用相应的操作。在大多数情况下，这意味着一个简单的 Namespace 对象将从命令行参数中解
# 析出的属性构建
#######################################################################################

args = parser.parse_args()
print(args.accumulate(args.integers))