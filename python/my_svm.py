import numpy as np
import sklearn.datasets as dsets


class SVM(object):
   def __init__(self, training_path, testing_path):
      self.training_path = training_path
      self.testing_path = testing_path
      self.training_x, self.training_y = dsets.load_svmlight_file(self.training_path)
      self.testing_x, self.testing_y = dsets.load_svmlight_file(self.testing_path)
      self.training_predict, self.testing_predict = np.zeros(self.training_y.shape), np.zeros(self.testing_y.shape)
      self.training_x, self.testing_x = self.training_x.toarray(), self.testing_x.toarray()

      self.training_x = self.training_x[:, :-1]
      assert self.training_x.shape[1]==self.testing_x.shape[1], "dimention not fit! %d, %d"%(self.training_x.shape[1], self.testing_x.shape[1])
      self.w = np.zeros((self.training_x.shape[1]+1,1))
      self.training_x = np.column_stack((np.ones((self.training_x.shape[0],1)), self.training_x))
      self.testing_x = np.column_stack((np.ones((self.testing_x.shape[0], 1)), self.testing_x))
      self.loss = []
      self.accuracy = []


   def train(self, lr=0.001, iters=1000000, C=10, batch_size=16):
      for i in range(iters):

         # get training samples
         if batch_size is None:
            x = self.training_x
            y = self.training_y
         else:
            sample_index = np.random.randint(self.training_x.shape[0], size=batch_size)
            x = self.training_x[sample_index]
            y = self.training_y[sample_index]


         xw_b = np.dot(x, self.w)
         xw_b = xw_b.reshape(y.shape)

         yxw_b = 1 - y*xw_b
         yxw_b = yxw_b.reshape((y.shape[0],1))
         eta = np.maximum(0, yxw_b)

         # L2norm
         # loss = np.dot(self.w.transpose(), self.w)/2 + C/2*np.power(eta, 2).sum()

         # compute L1norm
         if batch_size is None:
            # for gd
            loss = np.dot(self.w.transpose(), self.w)/2 + C*eta.sum()
         else:
            # for sgd
            loss = np.dot(self.w.transpose(), self.w)/2 + C*eta.mean()

         theta_w = np.zeros(self.w.shape)
         for j in range(yxw_b.shape[0]):
            if yxw_b[j] > 0:
               # gradient for L2Norm
               # theta_w += (-self.training_y[j]*(self.training_x[j].reshape(self.w.shape)))*yxw_b[j]

               # gradient for L1Norm
               theta_w += (-y[j]*(x[j].reshape(self.w.shape)))

         # compute mean if mode is sgd
         if batch_size is not None:
            theta_w /= 1.0*batch_size

         # update weights
         self.w = self.w - lr*(self.w +  C*theta_w)
         grad_norm = np.power(self.w +  C*theta_w, 2).sum()

         # save loss of every iterations
         self.loss.append(loss[0][0])

         # prediction and compute accuracy
         self.training_predict = np.sign(np.dot(self.training_x, self.w).reshape(self.training_y.shape))
         # print self.training_predict.shape
         accuracy = len(np.where((self.training_predict - self.training_y)==0)[0])*1.0/self.training_y.shape[0]

         self.accuracy.append(accuracy)
         print "#%d, loss: %f, accuracy: %f, grad_norm: %f" % (i, loss, accuracy, grad_norm)

      print "svm training done!"
      
