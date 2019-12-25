# coding=utf-8
"""
@author: liuyang
@time: 2019-10-30==13-49-34
@target: 
"""
# relation Distribution MM   引入依存句法信息,构建query
up_shapes = DependenceMM.shape[:-2]
# DependenceMM = DependenceMM.reshape((-1,*DependenceMM.shape[-2:]) )
# 这种[e]*n 再stack的用法可能有梯度没有传递到的问题.这里先满足维度运算 测试,暂且放下.
# 尽量换成下面的Linear,详细映射细节先不考虑
QueryforEntities_pre = self.QueryGeneratorAssitor1(DependenceMM)
# QueryforEntities_pre = DependenceMM.bmm(torch.stack([self.QueryGeneratorAssitor]*DependenceMM.shape[0]) )
QueryforEntities = self.QueryforEntitiesEmb1(QueryforEntities_pre)
# QueryforEntities = QueryforEntities_pre.bmm(torch.stack([self.QueryforEntitiesEmb]*DependenceMM.shape[0]) )
QueryforEntities = QueryforEntities.reshape((*up_shapes, *QueryforEntities.shape[-2 :]))
KeyforRelations = self.KeyforRelationsEmb(h)
RelationDistributionMM = torch.zeros((*QueryforEntities.shape[:-1], QueryforEntities.shape[-2]))



# relation Distribution MM   引入依存句法信息,构建query
# up_shapes = DependenceMM.shape[:-2]
# #这种[e]*n 再stack的用法可能有梯度没有传递到的问题.这里先满足维度运算 测试,暂且放下.
# #尽量换成下面的Linear,详细映射细节先不考虑
# QueryforEntities_pre = self.QueryGeneratorAssitor1(DependenceMM)
# QueryforEntities = self.QueryforEntitiesEmb1(QueryforEntities_pre)
# QueryforEntities = QueryforEntities.reshape((*up_shapes,*QueryforEntities.shape[-2:]))
# KeyforRelations = self.KeyforRelationsEmb(h)
# temp_batch = []
# for batch_idx,dim1right in enumerate(QueryforEntities):
#     temp_words=[]
#     for words_idx,oneWordQuery in enumerate(dim1right):
#         # temp_words.append( attention(oneWordQuery,KeyforRelations[batch_idx],distributionPolarFactors = nnMatrix[batch_idx][words_idx]*1.0 ) )
#         temp_words.append(attention(oneWordQuery, KeyforRelations[batch_idx]) )
#     temp_batch.append(torch.stack(temp_words))
# RelationDistributionMM = torch.stack(temp_batch)