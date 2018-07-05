import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';
import Table, { TableBody, TableCell, TableHead, TableRow } from 'material-ui/Table';
import Card, { CardActions, CardContent, CardMedia } from 'material-ui/Card';
import Typography from 'material-ui/Typography';
import ResultImg from './result.png';
import {Icon} from 'antd';

const styles = theme => ({
    container: {
      marginLeft: '5%',
      marginRight: '5%',
      marginTop: 20,
      marginBottom: 80,
    },
    textField: {
      marginLeft: theme.spacing.unit,
      marginRight: theme.spacing.unit,
    },
    comment: {
        marginTop:0
    },
    loading: {
      width: '100%',
      position: 'fixed',
      top: 1,
      left: 0
    },
    table_root: {
        width: '100%',
        marginTop: theme.spacing.unit * 3,
        overflowX: 'auto',
    },
    table: {
        minWidth: 400,
    },
    card: {
        width: '100%'
    },
    media: {
        width:'80%',
        marginLeft: '10%',
        height: 300,
    },
  });

const Iconstyles ={
  style:{
    width: "1em",
    height: "1em",
  }
 
};
let id = 0;
function createData(name, calories, fat, carbs, protein) {
    id += 1;
    return { id, name, calories, fat, carbs, protein };
  }
  
const data = [
    {model: 'SVM', time:284.4615},
    {model: 'LR', time:0.9675},
    {model: 'NB', time:0.0610}
];
  
class Intro extends React.Component {
  
    
    render() {
        const { classes } = this.props;
        return <div className={classes.container} >
            <h3>任务描述</h3>
            实现一个垃圾短信识别系统，在给定的数据集上验证效果
            
            <h3>数据概况</h3>
            <ul>
                <li>带标签数据（用于模型训练和测试）</li>
                <ol>
                    <li>标签域：1表示垃圾短信/0表示正常短信</li>
                    <li>文本域：短信源文本（进行了一些处理）</li>
                </ol>
                <li>不带标签数据（用于线上模拟）</li>
            </ul>

            <h3>数据集分割</h3>
            <div>
                随机分割100000条作为不同模型训练结果的比较集合
            </div>
            <div>
                5-foldcross validation交叉验证用于调参
            </div>
            

            <h3>模型训练</h3>
            <div>
                本系统使用SVM，Naive Bayes，Logistic Regression模型，比较了不同模型在测试集合上的Precision/Recall/F1的表现。
                其中，NB使用n-gram+tfidf作为特征，而LR和SVM使用Jieba切词后的词语的tfidf值为特征。
            </div>
            <div>
                此外，系统还随机对100条短信分类计算了不同模型的预测时长。
            </div>
            <h3>实验结果</h3>
            <h4>速度</h4>
            <div className={classes.table_root}>
                <Table className={classes.table}>
                    <TableHead>
                        <TableRow>
                            <TableCell>模型</TableCell>
                            <TableCell>SVM</TableCell>
                            <TableCell>LR</TableCell>
                            <TableCell>NB</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        <TableRow key={1}>
                            <TableCell>100条(s)</TableCell>
                            
                            {data.map(item=>{
                                return <TableCell numeric>{item.time.toFixed(2)}</TableCell>
                            })}
                        </TableRow>
                    </TableBody>
                </Table>
            </div>
            <h4>精度</h4>

            <Card className={classes.card}>
                <CardMedia
                    className={classes.media}
                    image={ResultImg}
                    title="Contemplative Reptile"
                />
                <CardContent style={{textAlign:'center'}}>
 
                <Typography component="p" className={classes.comment}>
                    其中NB使用n-gram（未分词），而LR和SVM使用Jieba分词
                </Typography>
                </CardContent>
                
            </Card>

            <h3>
                相关连接
            </h3>
            
            <h4><Icon type="github" />: <a href="https://github.com/h12345jack/webdm">webdm</a></h4>
            <h4><Icon type="link" />: <a href="http://scikit-learn.org/stable/">scikit-learn</a></h4>
            <h4><Icon type="github" />: <a href="https://github.com/fxsjy/jieba">jieba</a></h4>
            
         </div>

    }

}

export default withStyles(styles)(Intro);
