require(data.table)

submission = fread('E:/ML/ProjectSubmission-TeamX.csv')
colnames = names(submission)
result = fread('E:/ML/predictedrf10m.csv')
result = result[-1,]
result = result[,-1]
submission = submission[,-2]
final = cbind(submission, result)
colnames(final) = colnames
fwrite(final,'E:/ML/ProjectSubmission-Team6.csv')