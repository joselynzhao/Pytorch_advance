#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/8/22 下午7:24
# @Author  : Joselynzhao
# @Email   : zhaojing17@forxmail.com
# @File    : 101.py
# @Software: PyCharm
# @Desc    :

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isSymmetric(self, root) :
        # 中序遍历是对称的就行
        out = []
        def middle_bilian(root):
            if not root:
                print("execting")
                out.append(-1)
                return
            if root.left:
                middle_bilian(root.left)
            out.append(root.val)
            if root.right:
                middle_bilian(root.right)
        middle_bilian(root)
        print(out)
        for i in range(round(len(out)/2)):
            if out[i]!=out[len(out)-1-i]:
                return False
        return True

solu = Solution()
