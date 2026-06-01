(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,67271,e=>{"use strict";let t,r,n,a;e.i(20815);var s,i,o,l,u,h,d,p,c,f,m,g,x=e.i(68943),y=e.i(88951),b=e.i(53612),v=e.i(34111);let w={kernelName:x.Abs,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)(e,(0,v.step)((0,y.cast)(r,"float32"),-1))}}};var C=e.i(38796),I=e.i(20960),k=e.i(86144),S=e.i(82881),T=e.i(71199),N=e.i(1869);let $={kernelName:x.Acos,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>{let t=(0,T.square)((0,y.cast)(r,"float32")),n=(0,S.sqrt)((0,N.sub)((0,k.scalar)(1),t));return(0,I.neg)((0,C.div)(e,n))}}}},R={kernelName:x.Acosh,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>{let t=(0,S.sqrt)((0,N.sub)((0,T.square)((0,y.cast)(r,"float32")),1));return(0,C.div)(e,t)}}}};var A=e.i(23711),E=e.i(52397),F=e.i(30971);let D={kernelName:x.Add,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t,a=A.assertAndGetBroadcastShape(r.shape,n.shape);return{a:()=>{let t=e,n=A.getReductionAxes(r.shape,a);return n.length>0&&(t=(0,F.sum)(t,n)),(0,E.reshape)(t,r.shape)},b:()=>{let t=e,r=A.getReductionAxes(n.shape,a);return r.length>0&&(t=(0,F.sum)(t,r)),(0,E.reshape)(t,n.shape)}}}},O={kernelName:x.AddN,saveAllInputs:!0,gradFunc:(e,t)=>{let r={};return t.forEach((t,n)=>{r[n]=()=>e.clone()}),r}};var L=e.i(58879);let z={kernelName:x.ArgMax,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,L.zerosLike)(r)}}},_={kernelName:x.ArgMin,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,L.zerosLike)(r)}}},M={kernelName:x.Asin,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,C.div)(e,(0,S.sqrt)((0,N.sub)((0,k.scalar)(1),(0,T.square)((0,y.cast)(r,"float32")))))}}};var P=e.i(51735);let B={kernelName:x.Asinh,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>{let t=(0,S.sqrt)((0,P.add)((0,k.scalar)(1),(0,T.square)((0,y.cast)(r,"float32"))));return(0,C.div)(e,t)}}}},W={kernelName:x.Atan2,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t,a=(0,A.assertAndGetBroadcastShape)(r.shape,n.shape);return{a:()=>{let t=(0,P.add)((0,T.square)(r),(0,T.square)(n)),s=(0,b.mul)(e,(0,C.div)(n,t)),i=(0,A.getReductionAxes)(r.shape,a);return i.length>0&&(s=(0,F.sum)(s,i)),(0,E.reshape)(s,r.shape)},b:()=>{let t=(0,P.add)((0,T.square)(r),(0,T.square)(n)),s=(0,I.neg)((0,b.mul)(e,(0,C.div)(r,t))),i=(0,A.getReductionAxes)(n.shape,a);return i.length>0&&(s=(0,F.sum)(s,i)),(0,E.reshape)(s,n.shape)}}}},G={kernelName:x.Atan,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,C.div)(e,(0,P.add)((0,T.square)((0,y.cast)(r,"float32")),1))}}},U={kernelName:x.Atanh,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,C.div)(e,(0,N.sub)((0,k.scalar)(1),(0,T.square)((0,y.cast)(r,"float32"))))}}};var V=e.i(35898),H=e.i(61828),q=e.i(14219),j=e.i(20037),X=e.i(83084);let K=(0,X.op)({avgPool3dGrad_:function(e,t,r,n,a,s){let i=(0,H.convertToTensor)(e,"dy","avgPool3dGrad"),o=(0,H.convertToTensor)(t,"input","avgPool3dGrad"),l=i,u=o,h=!1;4===o.rank&&(h=!0,l=(0,E.reshape)(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]]),u=(0,E.reshape)(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),q.assert(5===l.rank,()=>`Error in avgPool3dGrad: dy must be rank 5 but got rank ${l.rank}.`),q.assert(5===u.rank,()=>`Error in avgPool3dGrad: input must be rank 5 but got rank ${u.rank}.`),(0,j.checkPadOnDimRoundingMode)("avgPool3dGrad",a,s);let d={dy:l,input:u},p=V.ENGINE.runKernel(x.AvgPool3DGrad,d,{filterSize:r,strides:n,pad:a,dimRoundingMode:s});return h?(0,E.reshape)(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}}),Y={kernelName:x.AvgPool3D,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{filterSize:a,strides:s,pad:i,dimRoundingMode:o}=r;return{x:()=>K(e,n,a,s,i,o)}}},Z=(0,X.op)({avgPoolGrad_:function(e,t,r,n,a){let s=(0,H.convertToTensor)(e,"dy","avgPoolGrad"),i=(0,H.convertToTensor)(t,"input","avgPoolGrad");q.assert(i.rank===s.rank,()=>`Rank of input (${i.rank}) does not match rank of dy (${s.rank})`);let o=i,l=s,u=!1;3===i.rank&&(u=!0,o=(0,E.reshape)(i,[1,i.shape[0],i.shape[1],i.shape[2]]),l=(0,E.reshape)(s,[1,s.shape[0],s.shape[1],s.shape[2]])),q.assert(4===l.rank,()=>`Error in avgPoolGrad: dy must be rank 4 but got rank ${l.rank}.`),q.assert(4===o.rank,()=>`Error in avgPoolGrad: input must be rank 4 but got rank ${o.rank}.`);let h={dy:l,input:o},d=V.ENGINE.runKernel(x.AvgPoolGrad,h,{filterSize:r,strides:n,pad:a});return u?(0,E.reshape)(d,[d.shape[1],d.shape[2],d.shape[3]]):d}}),J={kernelName:x.AvgPool,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{filterSize:a,strides:s,pad:i}=r;return{x:()=>Z(e,n,a,s,i)}}};var Q=e.i(45768);let ee={kernelName:x.BatchMatMul,inputsToSave:["a","b"],gradFunc:(e,t,r)=>{let[n,a]=t,{transposeA:s,transposeB:i}=r;return s||i?!s&&i?{a:()=>(0,Q.matMul)(e,a,!1,!1),b:()=>(0,Q.matMul)(e,n,!0,!1)}:s&&!i?{a:()=>(0,Q.matMul)(a,e,!1,!0),b:()=>(0,Q.matMul)(n,e,!1,!1)}:{a:()=>(0,Q.matMul)(a,e,!0,!0),b:()=>(0,Q.matMul)(e,n,!0,!0)}:{a:()=>(0,Q.matMul)(e,a,!1,!0),b:()=>(0,Q.matMul)(n,e,!0,!1)}}};var et=e.i(11435);let er={kernelName:x.BatchToSpaceND,gradFunc:(e,t,r)=>{let{blockShape:n,crops:a}=r;return{x:()=>(0,et.spaceToBatchND)(e,n,a)}}},en={kernelName:x.BroadcastTo,gradFunc:(e,t,r)=>{let n=r.inputShape,a=r.shape,s=Array.from(a);for(let e=n.length-1;e>=0;e--)if(n[e]===a[e])s[e]=1;else if(1!==n[e])throw Error(`broadcastTo(): [${n}] cannot be broadcast to [${a}].`);let i=[];for(let e=0;e<s.length;e++)s[e]>1&&i.push(e);return{x:()=>(0,F.sum)(e,i,!0)}}},ea={kernelName:x.Cast,gradFunc:e=>({x:()=>e.clone()})},es={kernelName:x.Ceil,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})};var ei=e.i(70208),eo=e.i(86157),el=e.i(50081),eu=e.i(77828);let eh={kernelName:x.ClipByValue,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{clipValueMin:a,clipValueMax:s}=r;return{x:()=>(0,eu.where)((0,el.logicalAnd)((0,ei.greaterEqual)(n,a),(0,eo.lessEqual)(n,s)),e,(0,L.zerosLike)(e))}}},ed={kernelName:x.ComplexAbs,inputsToSave:["x"],gradFunc:w.gradFunc};var ep=e.i(31907);let ec={kernelName:x.Concat,saveAllInputs:!0,gradFunc:(e,t,r)=>{let n=t.map(e=>e.shape),{axis:a}=r,s=(0,q.parseAxisParam)(a,t[0].shape)[0],i=n.map(e=>e[s]);return(0,ep.split)(e,i,s).map(e=>()=>e)}};var ef=e.i(71153),em=e.i(57293);let eg={kernelName:x.Conv2D,inputsToSave:["x","filter"],gradFunc:(e,t,r)=>{let[n,a]=t,{dilations:s,strides:i,pad:o,dataFormat:l}=r;return q.assert(j.tupleValuesAreOne(s),()=>`Error in gradient of conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${s}'`),{x:()=>(0,em.conv2DBackpropInput)(n.shape,e,a,i,o,l),filter:()=>(0,ef.conv2DBackpropFilter)(n,e,a.shape,i,o,l)}}};var ex=e.i(46546);let ey={kernelName:x.Conv2DBackpropInput,inputsToSave:["dy","filter"],gradFunc:(e,t,r)=>{let[n,a]=t,{strides:s,pad:i,dataFormat:o,dimRoundingMode:l}=r;return{dy:()=>(0,ex.conv2d)(e,a,s,i,o,1,l),filter:()=>(0,ef.conv2DBackpropFilter)(e,n,a.shape,s,i,o,l)}}},eb=(0,X.op)({conv3DBackpropFilter_:function(e,t,r,n,a){let s=e;4===e.rank&&(s=(0,E.reshape)(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]]));let i=t;4===i.rank&&(i=(0,E.reshape)(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]])),q.assert(5===s.rank,()=>`Error in conv3dDerFilter: input must be rank 5, but got shape ${s.shape}.`),q.assert(5===i.rank,()=>`Error in conv3dDerFilter: dy must be rank 5, but got shape ${i.shape}.`),q.assert(5===r.length,()=>`Error in conv3dDerFilter: filterShape must be length 5, but got ${r}.`),q.assert(s.shape[4]===r[3],()=>`Error in conv3dDerFilter: depth of input ${s.shape[4]}) must match input depth in filter (${r[3]}.`),q.assert(i.shape[4]===r[4],()=>`Error in conv3dDerFilter: depth of dy (${i.shape[4]}) must match output depth for filter (${r[4]}).`);let o={x:s,dy:i};return V.ENGINE.runKernel(x.Conv3DBackpropFilterV2,o,{strides:n,pad:a,filterShape:r})}});var ev=e.i(81489);let ew={kernelName:x.Conv3D,inputsToSave:["x","filter"],gradFunc:(e,t,r)=>{let{dilations:n,strides:a,pad:s}=r;q.assert((0,j.tupleValuesAreOne)(n),()=>`Error in gradient of conv3D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${n}'`);let[i,o]=t;return{x:()=>(0,ev.conv3DBackpropInput)(i.shape,e,o,a,s),filter:()=>eb(i,e,o.shape,a,s)}}};var eC=e.i(21593);let eI={kernelName:x.Cos,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)((0,I.neg)((0,eC.sin)((0,y.cast)(r,"float32"))),e)}}};var ek=e.i(43872);let eS={kernelName:x.Cosh,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)((0,ek.sinh)((0,y.cast)(r,"float32")),e)}}};var eT=e.i(80292),eN=e.i(95906),e$=e.i(94070);let eR={kernelName:x.Cumsum,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{axis:a,exclusive:s,reverse:i}=r;return{x:()=>{let t=(0,eT.getAxesPermutation)([a],n.rank),r=(0,eN.cumsum)(e,a,s,!i);return null!=t&&(r=(0,e$.transpose)(r,t)),r}}}};var eA=e.i(47153),eE=e.i(26541);let eF={kernelName:x.DepthwiseConv2dNative,inputsToSave:["x","filter"],gradFunc:(e,t,r)=>{let{dilations:n,strides:a,pad:s,dimRoundingMode:i}=r,o=null==n?[1,1]:n;q.assert(j.tupleValuesAreOne(o),()=>`Error in gradient of depthwiseConv2dNative: dilation rates greater than 1 are not yet supported. Got dilations '${o}'`);let[l,u]=t;return q.assert(4===l.rank,()=>`Error in gradient of depthwiseConv2dNative: input must be rank 4, but got rank ${l.rank}.`),q.assert(4===u.rank,()=>`Error in gradient of depthwiseConv2dNative: filter must be rank 4, but got rank ${u.rank}.`),q.assert(l.shape[3]===u.shape[2],()=>`Error in gradient of depthwiseConv2d: number of input channels (${l.shape[3]}) must match the inChannels dimension in filter ${u.shape[2]}.`),q.assert(j.eitherStridesOrDilationsAreOne(a,o),()=>`Error in gradient of depthwiseConv2d: Either strides or dilations must be  1. Got strides ${a} and dilations '${o}'.`),j.checkPadOnDimRoundingMode("depthwiseConv2d",s,i),{x:()=>(0,eE.depthwiseConv2dNativeBackpropInput)(l.shape,e,u,a,s,o,i),filter:()=>(0,eA.depthwiseConv2dNativeBackpropFilter)(l,e,u.shape,a,s,o,i)}}},eD={kernelName:x.Dilation2D,inputsToSave:["x","filter"],gradFunc:(e,t,r)=>{let[n,a]=t,s={x:n,filter:a,dy:e},i={x:n,filter:a,dy:e};return{x:()=>V.ENGINE.runKernel(x.Dilation2DBackpropInput,s,r),filter:()=>V.ENGINE.runKernel(x.Dilation2DBackpropFilter,i,r)}}},eO={kernelName:x.Elu,outputsToSave:[!0],gradFunc:(e,t)=>{let[r]=t,n={dy:e,y:r};return{x:()=>V.ENGINE.runKernel(x.EluGrad,n)}}};var eL=e.i(4774);let ez={kernelName:x.Erf,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t,n=(0,b.mul)((0,eL.exp)((0,I.neg)((0,T.square)(r))),2/Math.sqrt(Math.PI));return{x:()=>(0,b.mul)(e,n)}}},e_={kernelName:x.Exp,outputsToSave:[!0],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)(e,r)}}},eM={kernelName:x.ExpandDims,inputsToSave:["input"],gradFunc:(e,t)=>{let[r]=t;return{input:()=>(0,E.reshape)(e,r.shape)}}},eP={kernelName:x.Expm1,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)(e,(0,eL.exp)(r))}}},eB={kernelName:x.Floor,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})},eW={kernelName:x.FloorDiv,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t,a=(0,A.assertAndGetBroadcastShape)(r.shape,n.shape);return{a:()=>{let t=(0,C.div)(e,(0,y.cast)(n,"float32")),s=(0,A.getReductionAxes)(r.shape,a);return s.length>0?(0,E.reshape)((0,F.sum)(t,s),r.shape):t},b:()=>{let t=(0,b.mul)(e,(0,y.cast)(r,"float32")),s=(0,A.getReductionAxes)(n.shape,a);s.length>0&&(t=(0,E.reshape)((0,F.sum)(t,s),n.shape));let i=(0,T.square)(n);return(0,I.neg)((0,C.div)(t,(0,y.cast)(i,"float32")))}}}};var eG=e.i(94264),eU=e.i(92013);let eV={kernelName:x.FusedBatchNorm,inputsToSave:["x","mean","variance","scale"],gradFunc:(e,t,r)=>{let{varianceEpsilon:n}=r,[a,s,i,o]=t,l=null==o?(0,k.scalar)(1):o,u=(0,A.getReductionAxes)(s.shape,a.shape),h=[];if(1===s.rank){for(let e=0;e<a.shape.length-1;++e)h.push(a.shape[e]);h.push(1)}let d=(0,N.sub)(a,s),p=(0,b.mul)(e,l),c=(0,eG.rsqrt)((0,P.add)(i,(0,k.scalar)(n))),f=(0,b.mul)((0,b.mul)((0,b.mul)(c,c),c),(0,k.scalar)(-.5));return{x:()=>1===s.rank?(0,E.reshape)((0,b.mul)((0,b.mul)(e,(0,eU.tile)((0,E.reshape)(c,[1,1,1,s.shape[0]]),h)),l),a.shape):(0,E.reshape)((0,b.mul)((0,b.mul)(e,c),l),a.shape),mean:()=>{let e=(0,b.mul)((0,b.mul)(c,(0,k.scalar)(-1)),p);return 1===s.rank&&(e=(0,F.sum)(e,u)),(0,E.reshape)(e,s.shape)},variance:()=>{let e=(0,b.mul)((0,b.mul)(f,d),p);return 1===s.rank&&(e=(0,F.sum)(e,u)),(0,E.reshape)(e,s.shape)},scale:()=>{let t=(0,b.mul)(d,c),r=(0,b.mul)(e,t);return 1===s.rank&&(r=(0,F.sum)(r,u)),(0,E.reshape)(r,s.shape)},offset:()=>{let t=e;return 1===s.rank&&(t=(0,F.sum)(t,u)),(0,E.reshape)(t,s.shape)}}}};var eH=e.i(54478),eq=e.i(78188);let ej={kernelName:x.GatherV2,inputsToSave:["x","indices"],gradFunc:(e,t,r)=>{let[n,a]=t,{axis:s,batchDims:i}=r,o=(0,q.parseAxisParam)(s,n.shape)[0],l=(e,t,r)=>()=>{let n=e.shape,a=t.size,i=n.slice(0,o),l=i.length,u=n.slice(s,n.length).slice(1),h=u.length,d=eX(0,l),p=eX(l+1,l+1+h),c=eK([i,[a],u]),f=(0,E.reshape)(r,c),m=(0,E.reshape)(t,[a]),g=eK([[l],d,p]),x=(0,e$.transpose)(f,g),y=(0,eq.unsortedSegmentSum)(x,m,e.shape[o]),b=(0,eT.getUndoAxesPermutation)(g);return(0,e$.transpose)(y,b)};if(1!==i)return{x:l(n,a,e),indices:()=>a};{let t=n.shape[0],r=n.split(t,0);return{x:()=>(0,eH.stack)(r.map((t,r)=>l(t,a.slice(r,1),e.slice(r,1))())).reshape(n.shape),indices:()=>a}}}};function eX(e,t){let r=[];for(let n=e;n<t;++n)r.push(n);return r}function eK(e){let t=[];for(let r=0;r<e.length;++r)for(let n=0;n<e[r].length;++n)t.push(e[r][n]);return t}let eY={kernelName:x.GreaterEqual,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t;return{a:()=>(0,L.zerosLike)(r),b:()=>(0,L.zerosLike)(n)}}},eZ={kernelName:x.Identity,gradFunc:e=>({x:()=>(0,y.cast)(e,"float32")})},eJ={kernelName:x.IsFinite,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})},eQ={kernelName:x.IsInf,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})},e0={kernelName:x.IsNan,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})};var e1=e.i(77043);let e2={kernelName:x.LeakyRelu,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{alpha:a}=r,s=(0,e1.greater)(n,0);return{x:()=>(0,eu.where)(s,e,(0,b.mul)(e,a))}}},e3={kernelName:x.Log1p,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,C.div)(e,(0,P.add)(r,1))}}},e4={kernelName:x.Log,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,C.div)(e,(0,y.cast)(r,"float32"))}}},e5={kernelName:x.LogSoftmax,inputsToSave:[],outputsToSave:[!0],gradFunc:(e,t,r)=>{let[n]=t,{axis:a}=r;return{logits:()=>{let t=(0,eL.exp)(n);return(0,N.sub)(e,(0,b.mul)((0,F.sum)(e,a,!0),t))}}}},e6=(0,X.op)({localResponseNormalizationBackprop_:function(e,t,r,n=5,a=1,s=1,i=.5){return V.ENGINE.runKernel(x.LRNGrad,{x:e,y:t,dy:r},{depthRadius:n,bias:a,alpha:s,beta:i})}}),e8={kernelName:x.LRN,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,r)=>{let[n,a]=t,{depthRadius:s,bias:i,alpha:o,beta:l}=r;return{x:()=>e6(n,a,e,s,i,o,l)}}};var e7=e.i(6022);function e9(e,t,r,n){return t.rank<r.rank&&(t=(0,E.reshape)(t,eT.expandShapeToKeepDim(t.shape,n))),e.rank<r.rank&&(e=(0,E.reshape)(e,eT.expandShapeToKeepDim(e.shape,n))),{x:()=>(0,b.mul)(e,(0,y.cast)((0,e7.equal)(r,t),e.dtype))}}let te={kernelName:x.Max,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,r)=>{let{reductionIndices:n}=r,a=t[0],s=t[1],i=q.parseAxisParam(n,a.shape),o=e9(e,s,a,i);return{x:()=>o.x()}}};var tt=e.i(31642);let tr={kernelName:x.Maximum,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t;return{a:()=>(0,b.mul)(e,(0,y.cast)((0,ei.greaterEqual)(r,n),"float32")),b:()=>(0,b.mul)(e,(0,y.cast)((0,tt.less)(r,n),"float32"))}}},tn=(0,X.op)({maxPool3dGrad_:function(e,t,r,n,a,s,i){let o=(0,H.convertToTensor)(e,"dy","maxPool3dGrad"),l=(0,H.convertToTensor)(t,"input","maxPool3dGrad"),u=(0,H.convertToTensor)(r,"output","maxPool3dGrad"),h=o,d=l,p=u,c=!1;4===l.rank&&(c=!0,h=(0,E.reshape)(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]]),d=(0,E.reshape)(l,[1,l.shape[0],l.shape[1],l.shape[2],l.shape[3]]),p=(0,E.reshape)(u,[1,u.shape[0],u.shape[1],u.shape[2],u.shape[3]])),q.assert(5===h.rank,()=>`Error in maxPool3dGrad: dy must be rank 5 but got rank ${h.rank}.`),q.assert(5===d.rank,()=>`Error in maxPool3dGrad: input must be rank 5 but got rank ${d.rank}.`),q.assert(5===p.rank,()=>`Error in maxPool3dGrad: output must be rank 5 but got rank ${p.rank}.`),(0,j.checkPadOnDimRoundingMode)("maxPool3dGrad",s,i);let f={dy:h,input:d,output:p},m=V.ENGINE.runKernel(x.MaxPool3DGrad,f,{filterSize:n,strides:a,pad:s,dimRoundingMode:i});return c?(0,E.reshape)(m,[m.shape[1],m.shape[2],m.shape[3],m.shape[4]]):m}}),ta={kernelName:x.MaxPool3D,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,r)=>{let[n,a]=t,{filterSize:s,strides:i,pad:o,dimRoundingMode:l}=r;return{x:()=>tn(e,n,a,s,i,o,l)}}},ts=(0,X.op)({maxPoolGrad_:function(e,t,r,n,a,s,i){let o=(0,H.convertToTensor)(e,"dy","maxPoolGrad"),l=(0,H.convertToTensor)(t,"input","maxPoolGrad"),u=(0,H.convertToTensor)(r,"output","maxPoolGrad");return q.assert(l.rank===o.rank,()=>`Rank of input (${l.rank}) does not match rank of dy (${o.rank})`),q.assert(4===o.rank,()=>`Error in maxPoolGrad: dy must be rank 4 but got rank ${o.rank}.`),q.assert(4===l.rank,()=>`Error in maxPoolGrad: input must be rank 4 but got rank ${l.rank}.`),j.checkPadOnDimRoundingMode("maxPoolGrad",s,i),V.ENGINE.runKernel(x.MaxPoolGrad,{dy:o,input:l,output:u},{filterSize:n,strides:a,pad:s,dimRoundingMode:i})}}),ti={kernelName:x.MaxPool,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,r)=>{let[n,a]=t,{filterSize:s,strides:i,pad:o}=r;return{x:()=>ts(e,n,a,s,i,o)}}};var to=e.i(64539);let tl={kernelName:x.Mean,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{axis:a}=r,s=q.parseAxisParam(a,n.shape),i=(0,eT.computeOutAndReduceShapes)(n.shape,s)[1],o=q.sizeFromShape(i);return{x:()=>{let t=n.shape.slice();s.forEach(e=>{t[e]=1});let r=(0,E.reshape)(e,t);return(0,C.div)((0,b.mul)(r,(0,to.ones)(n.shape,"float32")),o)}}}},tu={kernelName:x.Min,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(e,t,r)=>{let{axis:n}=r,[a,s]=t,i=q.parseAxisParam(n,a.shape),o=e9(e,s,a,i);return{x:()=>o.x()}}},th={kernelName:x.Minimum,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t;return{a:()=>(0,b.mul)(e,(0,y.cast)((0,eo.lessEqual)(r,n),"float32")),b:()=>(0,b.mul)(e,(0,y.cast)((0,e1.greater)(r,n),"float32"))}}};var td=e.i(83507);let tp={kernelName:x.MirrorPad,inputsToSave:["x"],gradFunc:(e,t,r)=>{let n=t[0],{paddings:a}=r,s=a.map(e=>e[0]);return{x:()=>(0,td.slice)(e,s,n.shape)}}};var tc=e.i(35671);let tf={kernelName:x.Mod,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t,a=(0,A.assertAndGetBroadcastShape)(r.shape,n.shape);return{a:()=>{let t=(0,A.getReductionAxes)(r.shape,a);return t.length>0?(0,E.reshape)((0,F.sum)(e,t),r.shape):e},b:()=>{let t=(0,b.mul)(e,(0,I.neg)((0,tc.floor)((0,C.div)(r,n)))),s=(0,A.getReductionAxes)(n.shape,a);return s.length>0?(0,E.reshape)((0,F.sum)(t,s),n.shape):t}}}},tm={kernelName:x.Multiply,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t,a=(0,A.assertAndGetBroadcastShape)(r.shape,n.shape);return{a:()=>{let t=(0,b.mul)(e,(0,y.cast)(n,"float32")),s=(0,A.getReductionAxes)(r.shape,a);return s.length>0?(0,E.reshape)((0,F.sum)(t,s),r.shape):t},b:()=>{let t=(0,b.mul)(e,(0,y.cast)(r,"float32")),s=(0,A.getReductionAxes)(n.shape,a);return s.length>0?(0,E.reshape)((0,F.sum)(t,s),n.shape):t}}}},tg={kernelName:x.Neg,gradFunc:e=>({x:()=>(0,I.neg)(e)})};var tx=e.i(27665);let ty={kernelName:x.OneHot,inputsToSave:["indices"],gradFunc:(e,t)=>{let r=t[0];return{indices:()=>(0,tx.zeros)(r.shape,"float32")}}},tb={kernelName:x.OnesLike,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})};var tv=e.i(92781);let tw={kernelName:x.Pack,saveAllInputs:!0,gradFunc:(e,t,r)=>{let{axis:n}=r;return(0,tv.unstack)(e,n).map(e=>()=>e)}},tC={kernelName:x.PadV2,inputsToSave:["x"],gradFunc:(e,t,r)=>{let n=t[0],{paddings:a}=r,s=a.map(e=>e[0]);return{x:()=>(0,td.slice)(e,s,n.shape)}}};var tI=e.i(19959),tk=e.i(67639);let tS={kernelName:x.Pow,inputsToSave:["a","b"],outputsToSave:[!0],gradFunc:(e,t)=>{let[r,n,a]=t,s=A.assertAndGetBroadcastShape(r.shape,n.shape);return{a:()=>{let t=(0,y.cast)(n,"float32"),a=(0,b.mul)(e,(0,b.mul)(t,(0,tk.pow)(r,(0,N.sub)(t,(0,k.scalar)(1))))),i=A.getReductionAxes(r.shape,s);return i.length>0&&(a=(0,F.sum)(a,i)),(0,E.reshape)(a,r.shape)},b:()=>{let t=(0,e1.greater)(r,0),i=(0,eu.where)(t,(0,tI.log)(r),(0,L.zerosLike)(r)),o=(0,b.mul)(e,(0,b.mul)(a,i)),l=A.getReductionAxes(n.shape,s);return l.length>0&&(o=(0,F.sum)(o,l)),(0,E.reshape)(o,n.shape)}}}},tT={kernelName:x.Prelu,inputsToSave:["x","alpha"],gradFunc:(e,t)=>{let[r,n]=t,a=(0,e1.greater)(r,0);return{x:()=>(0,eu.where)(a,e,(0,b.mul)(e,n)),alpha:()=>{let t=(0,eu.where)(a,(0,L.zerosLike)(e),(0,b.mul)(e,r)),s=(0,A.getReductionAxes)(n.shape,e.shape);return s.length>0&&(t=(0,F.sum)(t,s)),(0,E.reshape)(t,n.shape)}}}};var tN=e.i(68647),tN=tN,t$=e.i(33800);let tR={kernelName:x.Prod,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{axis:a}=r,s=[];return s=null==a?n.shape.map((e,t)=>t):"number"==typeof a?[a]:a,{x:()=>(function(e,t,r){var n;let a,s,i,o,l,u=e.shape.length,h=u-r.length,d=tN.getAxesPermutation(r,u),p=e;null!=d&&(p=(0,e$.transpose)(e,d));let c=p.shape.slice(),f=c.splice(u-r.length,r.length).reduce((e,t)=>e*t,1);c.push(f);let m=(n=p.reshape(c),(a=n.shape.slice())[h]=1,s=(0,E.reshape)(t,a),i=(0,t$.cumprod)(n,h,!0,!1),o=(0,t$.cumprod)(n,h,!0,!0),l=(0,b.mul)(i,o),(0,b.mul)(s,l));if(m=m.reshape(p.shape),null!=d){let e=tN.getUndoAxesPermutation(d);m=(0,e$.transpose)(m,e)}return m})(n,e,s)}}},tA={kernelName:x.RealDiv,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t,a=A.assertAndGetBroadcastShape(r.shape,n.shape);return{a:()=>{let t=(0,C.div)(e,(0,y.cast)(n,"float32")),s=A.getReductionAxes(r.shape,a);return s.length>0?(0,E.reshape)((0,F.sum)(t,s),r.shape):t},b:()=>{let t=(0,b.mul)(e,(0,y.cast)(r,"float32")),s=A.getReductionAxes(n.shape,a);s.length>0&&(t=(0,E.reshape)((0,F.sum)(t,s),n.shape));let i=(0,T.square)(n);return(0,I.neg)((0,C.div)(t,(0,y.cast)(i,"float32")))}}}},tE={kernelName:x.Reciprocal,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,C.div)(e,(0,I.neg)((0,T.square)(r)))}}},tF={kernelName:x.Relu6,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t,n=(0,b.mul)((0,eo.lessEqual)(r,6),(0,v.step)(r));return{x:()=>(0,b.mul)(e,(0,y.cast)(n,"float32"))}}},tD={kernelName:x.Relu,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)(e,(0,y.cast)((0,v.step)(r),"float32"))}}},tO={kernelName:x.Reshape,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,E.reshape)(e,r.shape)}}},tL={kernelName:x.ResizeBilinear,inputsToSave:["images"],gradFunc:(e,t,r)=>{let[n]=t,a={dy:e,images:n};return{images:()=>V.ENGINE.runKernel(x.ResizeBilinearGrad,a,r)}}},tz={kernelName:x.ResizeNearestNeighbor,inputsToSave:["images"],gradFunc:(e,t,r)=>{let[n]=t,a={dy:e,images:n};return{images:()=>V.ENGINE.runKernel(x.ResizeNearestNeighborGrad,a,r)}}};var t_=e.i(35831);let tM={kernelName:x.Reverse,gradFunc:(e,t,r)=>{let{dims:n}=r,a=(0,q.parseAxisParam)(n,e.shape);return{x:()=>(0,t_.reverse)(e,a)}}},tP={kernelName:x.Round,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})},tB={kernelName:x.Rsqrt,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,I.neg)((0,C.div)(e,(0,b.mul)((0,tk.pow)(r,1.5),2)))}}};var tW=e.i(92555);let tG={kernelName:x.Select,inputsToSave:["condition"],gradFunc:(e,t)=>{let[r]=t;return{condition:()=>(0,y.cast)((0,L.zerosLike)(r),"float32"),t:()=>(0,b.mul)(e,(0,y.cast)(r,e.dtype)),e:()=>(0,b.mul)(e,(0,y.cast)((0,tW.logicalNot)(r),e.dtype))}}};var tU=e.i(47060);let tV={kernelName:x.Selu,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>{let t=(0,e1.greater)(r,(0,k.scalar)(0)),n=(0,k.scalar)(tU.SELU_SCALEALPHA),a=(0,k.scalar)(tU.SELU_SCALE),s=(0,b.mul)(e,a),i=(0,b.mul)((0,b.mul)(e,n),(0,eL.exp)((0,y.cast)(r,"float32")));return(0,eu.where)(t,s,i)}}}},tH={kernelName:x.Sigmoid,outputsToSave:[!0],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)(e,(0,b.mul)(r,(0,N.sub)((0,k.scalar)(1),r)))}}},tq={kernelName:x.Sign,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})};var tj=e.i(74755);let tX={kernelName:x.Sin,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)((0,tj.cos)((0,y.cast)(r,"float32")),e)}}};var tK=e.i(54012);let tY={kernelName:x.Sinh,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)((0,tK.cosh)((0,y.cast)(r,"float32")),e)}}};var tZ=e.i(73851),tJ=e.i(53298);let tQ={kernelName:x.Slice,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{begin:a,size:s}=r,i=n.shape,[o,l]=(0,tJ.parseSliceParams)(n,a,s),u=[];for(let t=0;t<e.rank;t++)u.push([o[t],i[t]-o[t]-l[t]]);return{x:()=>(0,tZ.pad)(e,u)}}},t0={kernelName:x.Softmax,outputsToSave:[!0],gradFunc:(e,t,r)=>{let[n]=t,{dim:a}=r,s=(0,b.mul)(e,n);return{logits:()=>(0,N.sub)(s,(0,b.mul)((0,F.sum)(s,[a],!0),n))}}};var t1=e.i(7721);let t2={kernelName:x.Softplus,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)(e,(0,t1.sigmoid)(r))}}};var t3=e.i(76914);let t4={kernelName:x.SpaceToBatchND,gradFunc:(e,t,r)=>{let{blockShape:n,paddings:a}=r;return{x:()=>(0,t3.batchToSpaceND)(e,n,a)}}};var t5=e.i(61540);let t6={kernelName:x.SplitV,gradFunc:(e,t,r)=>{let{axis:n}=r;return{x:()=>(0,t5.concat)(e,n)}}},t8={kernelName:x.Sqrt,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,C.div)(e,(0,b.mul)((0,S.sqrt)((0,y.cast)(r,"float32")),2))}}},t7={kernelName:x.Square,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)(e,(0,b.mul)((0,y.cast)(r,"float32"),2))}}},t9={kernelName:x.SquaredDifference,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t,a=(0,k.scalar)(2);return{a:()=>(0,b.mul)(e,(0,b.mul)(a,(0,N.sub)(r,n))),b:()=>(0,b.mul)(e,(0,b.mul)(a,(0,N.sub)(n,r)))}}},re={kernelName:x.Step,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})},rt={kernelName:x.Sub,inputsToSave:["a","b"],gradFunc:(e,t)=>{let[r,n]=t,a=A.assertAndGetBroadcastShape(r.shape,n.shape);return{a:()=>{let t=e,n=A.getReductionAxes(r.shape,a);return n.length>0&&(t=(0,F.sum)(t,n)),(0,E.reshape)(t,r.shape)},b:()=>{let t=e,r=A.getReductionAxes(n.shape,a);return r.length>0&&(t=(0,F.sum)(t,r)),(0,E.reshape)((0,I.neg)(t),n.shape)}}}},rr={kernelName:x.Sum,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,a=n.shape.slice(),{axis:s}=r;(0,q.parseAxisParam)(s,n.shape).forEach(e=>{a[e]=1});let i=(0,E.reshape)(e,a),o=(0,b.mul)(i,(0,to.ones)(n.shape,"float32"));return{x:()=>o}}},rn={kernelName:x.Tan,inputsToSave:["x"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,C.div)(e,(0,T.square)((0,tj.cos)(r)))}}},ra={kernelName:x.Tanh,outputsToSave:[!0],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(0,b.mul)((0,N.sub)((0,k.scalar)(1),(0,T.square)(r)),e)}}},rs={kernelName:x.Tile,inputsToSave:["x"],gradFunc:(e,t,r)=>{let[n]=t,{reps:a}=r;return{x:()=>{let t=(0,L.zerosLike)(n);if(1===n.rank)for(let r=0;r<a[0];++r)t=(0,P.add)(t,(0,td.slice)(e,[r*n.shape[0]],[n.shape[0]]));else if(2===n.rank)for(let r=0;r<a[0];++r)for(let s=0;s<a[1];++s)t=(0,P.add)(t,(0,td.slice)(e,[r*n.shape[0],s*n.shape[1]],[n.shape[0],n.shape[1]]));else if(3===n.rank)for(let r=0;r<a[0];++r)for(let s=0;s<a[1];++s)for(let i=0;i<a[2];++i)t=(0,P.add)(t,(0,td.slice)(e,[r*n.shape[0],s*n.shape[1],i*n.shape[2]],[n.shape[0],n.shape[1],n.shape[2]]));else if(4===n.rank)for(let r=0;r<a[0];++r)for(let s=0;s<a[1];++s)for(let i=0;i<a[2];++i)for(let o=0;o<a[3];++o)t=(0,P.add)(t,(0,td.slice)(e,[r*n.shape[0],s*n.shape[1],i*n.shape[2],o*n.shape[3]],[n.shape[0],n.shape[1],n.shape[2],n.shape[3]]));else throw Error(`Gradient for tile operation is not implemented for rank-${n.rank} tensors yet.`);return t}}}},ri={kernelName:x.Transpose,gradFunc:(e,t,r)=>{let{perm:n}=r,a=eT.getUndoAxesPermutation(n);return{x:()=>(0,e$.transpose)(e,a)}}},ro={kernelName:x.Unpack,gradFunc:(e,t,r)=>{let{axis:n}=r;return{value:()=>(0,eH.stack)(e,n)}}};var rl=e.i(31357),ru=e.i(83445),rh=e.i(81944);let rd={kernelName:x.UnsortedSegmentSum,inputsToSave:["segmentIds"],gradFunc:(e,t)=>{let[r]=t;return{x:()=>(function(e,t){let r=(0,rh.maximum)(t,(0,L.zerosLike)(t)),n=(0,ru.gather)(e,r),a=(0,ei.greaterEqual)(t,(0,k.scalar)(0,"int32")),s=n.rank-a.rank;for(let e=0;e<s;++e)a=(0,rl.expandDims)(a,e+1);a=(0,el.logicalAnd)(a,(0,to.ones)(n.shape,"bool"));let i=(0,L.zerosLike)(n);return(0,eu.where)(a,n,i)})(e,r)}}},rp={kernelName:x.ZerosLike,gradFunc:e=>({x:()=>(0,L.zerosLike)(e)})};var rc=e.i(17281);for(let e of[w,$,R,D,O,z,_,M,B,W,G,U,Y,J,ee,er,en,ea,es,eh,ed,ec,ey,eg,ew,eI,eS,eR,eF,eD,tA,eO,ez,e_,eM,eP,eW,eB,eV,ej,eY,eZ,eJ,eQ,e0,e2,e3,e4,e5,e8,te,te,tr,ta,ti,tl,tu,th,tp,tf,tm,tg,ty,tb,tw,tC,tC,tS,tT,tR,tE,tF,tD,tO,tL,tz,tM,tP,tB,tG,tV,tH,tq,tX,tY,tQ,t0,t2,t4,t4,t6,t6,t8,t9,t7,re,rt,rr,rn,ra,rs,ri,ro,rd,rp])(0,rc.registerGradient)(e);var rf=e.i(72393),rm=e.i(41365);(0,rm.getGlobalTensorClass)().prototype.abs=function(){return this.throwIfDisposed(),(0,rf.abs)(this)};var rg=e.i(36195);(0,rm.getGlobalTensorClass)().prototype.acos=function(){return this.throwIfDisposed(),(0,rg.acos)(this)};var rx=e.i(1262);(0,rm.getGlobalTensorClass)().prototype.acosh=function(){return this.throwIfDisposed(),(0,rx.acosh)(this)},(0,rm.getGlobalTensorClass)().prototype.add=function(e){return this.throwIfDisposed(),(0,P.add)(this,e)};var ry=e.i(64486);(0,rm.getGlobalTensorClass)().prototype.all=function(e,t){return this.throwIfDisposed(),(0,ry.all)(this,e,t)};var rb=e.i(40816);(0,rm.getGlobalTensorClass)().prototype.any=function(e,t){return this.throwIfDisposed(),(0,rb.any)(this,e,t)};var rv=e.i(6957);(0,rm.getGlobalTensorClass)().prototype.argMax=function(e){return this.throwIfDisposed(),(0,rv.argMax)(this,e)};var rw=e.i(21508);(0,rm.getGlobalTensorClass)().prototype.argMin=function(e){return this.throwIfDisposed(),(0,rw.argMin)(this,e)},(0,rm.getGlobalTensorClass)().prototype.asScalar=function(){return this.throwIfDisposed(),(0,q.assert)(1===this.size,()=>"The array must have only 1 element."),(0,E.reshape)(this,[])},(0,rm.getGlobalTensorClass)().prototype.asType=function(e){return this.throwIfDisposed(),(0,y.cast)(this,e)},(0,rm.getGlobalTensorClass)().prototype.as1D=function(){return this.throwIfDisposed(),(0,E.reshape)(this,[this.size])},(0,rm.getGlobalTensorClass)().prototype.as2D=function(e,t){return this.throwIfDisposed(),(0,E.reshape)(this,[e,t])},(0,rm.getGlobalTensorClass)().prototype.as3D=function(e,t,r){return this.throwIfDisposed(),(0,E.reshape)(this,[e,t,r])},(0,rm.getGlobalTensorClass)().prototype.as4D=function(e,t,r,n){return this.throwIfDisposed(),(0,E.reshape)(this,[e,t,r,n])},(0,rm.getGlobalTensorClass)().prototype.as5D=function(e,t,r,n,a){return this.throwIfDisposed(),(0,E.reshape)(this,[e,t,r,n,a])};var rC=e.i(52020);(0,rm.getGlobalTensorClass)().prototype.asin=function(){return this.throwIfDisposed(),(0,rC.asin)(this)};var rI=e.i(69637);(0,rm.getGlobalTensorClass)().prototype.asinh=function(){return this.throwIfDisposed(),(0,rI.asinh)(this)};var rk=e.i(62400);(0,rm.getGlobalTensorClass)().prototype.atan=function(){return this.throwIfDisposed(),(0,rk.atan)(this)};var rS=e.i(63395);(0,rm.getGlobalTensorClass)().prototype.atan2=function(e){return this.throwIfDisposed(),(0,rS.atan2)(this,e)};var rT=e.i(97679);(0,rm.getGlobalTensorClass)().prototype.atanh=function(){return this.throwIfDisposed(),(0,rT.atanh)(this)};var rN=e.i(34014);(0,rm.getGlobalTensorClass)().prototype.avgPool=function(e,t,r,n){return this.throwIfDisposed(),(0,rN.avgPool)(this,e,t,r,n)},(0,rm.getGlobalTensorClass)().prototype.batchToSpaceND=function(e,t){return this.throwIfDisposed(),(0,t3.batchToSpaceND)(this,e,t)};var r$=e.i(19301);(0,rm.getGlobalTensorClass)().prototype.batchNorm=function(e,t,r,n,a){return this.throwIfDisposed(),(0,r$.batchNorm)(this,e,t,r,n,a)};var rR=e.i(72366);(0,rm.getGlobalTensorClass)().prototype.broadcastTo=function(e){return this.throwIfDisposed(),(0,rR.broadcastTo)(this,e)},(0,rm.getGlobalTensorClass)().prototype.cast=function(e){return this.throwIfDisposed(),(0,y.cast)(this,e)};var rA=e.i(75535);(0,rm.getGlobalTensorClass)().prototype.ceil=function(){return this.throwIfDisposed(),(0,rA.ceil)(this)};var rE=e.i(46793);(0,rm.getGlobalTensorClass)().prototype.clipByValue=function(e,t){return this.throwIfDisposed(),(0,rE.clipByValue)(this,e,t)},(0,rm.getGlobalTensorClass)().prototype.concat=function(e,t){return this.throwIfDisposed(),e instanceof rm.Tensor&&(e=[e]),(0,t5.concat)([this,...e],t)};var rF=e.i(7308);(0,rm.getGlobalTensorClass)().prototype.conv1d=function(e,t,r,n,a,s){return this.throwIfDisposed(),(0,rF.conv1d)(this,e,t,r,n,a,s)};var rD=e.i(26373);(0,rm.getGlobalTensorClass)().prototype.conv2dTranspose=function(e,t,r,n,a){return this.throwIfDisposed(),(0,rD.conv2dTranspose)(this,e,t,r,n,a)},(0,rm.getGlobalTensorClass)().prototype.conv2d=function(e,t,r,n,a,s){return this.throwIfDisposed(),(0,ex.conv2d)(this,e,t,r,n,a,s)},(0,rm.getGlobalTensorClass)().prototype.cos=function(){return this.throwIfDisposed(),(0,tj.cos)(this)},(0,rm.getGlobalTensorClass)().prototype.cosh=function(){return this.throwIfDisposed(),(0,tK.cosh)(this)},(0,rm.getGlobalTensorClass)().prototype.cumprod=function(e,t,r){return this.throwIfDisposed(),(0,t$.cumprod)(this,e,t,r)},(0,rm.getGlobalTensorClass)().prototype.cumsum=function(e,t,r){return this.throwIfDisposed(),(0,eN.cumsum)(this,e,t,r)};var rO=e.i(25014);(0,rm.getGlobalTensorClass)().prototype.depthToSpace=function(e,t){return this.throwIfDisposed(),(0,rO.depthToSpace)(this,e,t)};var rL=e.i(55445);(0,rm.getGlobalTensorClass)().prototype.depthwiseConv2d=function(e,t,r,n,a,s){return this.throwIfDisposed(),(0,rL.depthwiseConv2d)(this,e,t,r,n,a,s)};var rz=e.i(79582);(0,rm.getGlobalTensorClass)().prototype.dilation2d=function(e,t,r,n,a){return this.throwIfDisposed(),(0,rz.dilation2d)(this,e,t,r,n,a)};var r_=e.i(99513);(0,rm.getGlobalTensorClass)().prototype.divNoNan=function(e){return this.throwIfDisposed(),(0,r_.divNoNan)(this,e)},(0,rm.getGlobalTensorClass)().prototype.div=function(e){return this.throwIfDisposed(),(0,C.div)(this,e)};var rM=e.i(45589);(0,rm.getGlobalTensorClass)().prototype.dot=function(e){return this.throwIfDisposed(),(0,rM.dot)(this,e)};var rP=e.i(29608);(0,rm.getGlobalTensorClass)().prototype.elu=function(){return this.throwIfDisposed(),(0,rP.elu)(this)},(0,rm.getGlobalTensorClass)().prototype.equal=function(e){return this.throwIfDisposed(),(0,e7.equal)(this,e)};var rB=e.i(16253);(0,rm.getGlobalTensorClass)().prototype.erf=function(){return this.throwIfDisposed(),(0,rB.erf)(this)};var rW=e.i(91822);(0,rm.getGlobalTensorClass)().prototype.euclideanNorm=function(e,t){return this.throwIfDisposed(),(0,rW.euclideanNorm)(this,e,t)},(0,rm.getGlobalTensorClass)().prototype.exp=function(){return this.throwIfDisposed(),(0,eL.exp)(this)},(0,rm.getGlobalTensorClass)().prototype.expandDims=function(e){return this.throwIfDisposed(),(0,rl.expandDims)(this,e)};var rG=e.i(54412);(0,rm.getGlobalTensorClass)().prototype.expm1=function(){return this.throwIfDisposed(),(0,rG.expm1)(this)};var rU=e.i(31326);(0,rm.getGlobalTensorClass)().prototype.fft=function(){return this.throwIfDisposed(),(0,rU.fft)(this)},(0,rm.getGlobalTensorClass)().prototype.flatten=function(){return this.throwIfDisposed(),(0,E.reshape)(this,[this.size])},(0,rm.getGlobalTensorClass)().prototype.floor=function(){return this.throwIfDisposed(),(0,tc.floor)(this)};var rV=e.i(76453);(0,rm.getGlobalTensorClass)().prototype.floorDiv=function(e){return this.throwIfDisposed(),(0,rV.floorDiv)(this,e)},(0,rm.getGlobalTensorClass)().prototype.gather=function(e,t,r){return this.throwIfDisposed(),(0,ru.gather)(this,e,t,r)},(0,rm.getGlobalTensorClass)().prototype.greaterEqual=function(e){return this.throwIfDisposed(),(0,ei.greaterEqual)(this,e)},(0,rm.getGlobalTensorClass)().prototype.greater=function(e){return this.throwIfDisposed(),(0,e1.greater)(this,e)};var rH=e.i(20397);(0,rm.getGlobalTensorClass)().prototype.ifft=function(){return this.throwIfDisposed(),(0,rH.ifft)(this)};var rq=e.i(29954);(0,rm.getGlobalTensorClass)().prototype.irfft=function(){return this.throwIfDisposed(),(0,rq.irfft)(this)};var rj=e.i(43569);(0,rm.getGlobalTensorClass)().prototype.isFinite=function(){return this.throwIfDisposed(),(0,rj.isFinite)(this)};var rX=e.i(7115);(0,rm.getGlobalTensorClass)().prototype.isInf=function(){return this.throwIfDisposed(),(0,rX.isInf)(this)};var rK=e.i(594);(0,rm.getGlobalTensorClass)().prototype.isNaN=function(){return this.throwIfDisposed(),(0,rK.isNaN)(this)};var rY=e.i(33370);(0,rm.getGlobalTensorClass)().prototype.leakyRelu=function(e){return this.throwIfDisposed(),(0,rY.leakyRelu)(this,e)},(0,rm.getGlobalTensorClass)().prototype.lessEqual=function(e){return this.throwIfDisposed(),(0,eo.lessEqual)(this,e)},(0,rm.getGlobalTensorClass)().prototype.less=function(e){return this.throwIfDisposed(),(0,tt.less)(this,e)};var rZ=e.i(73771);(0,rm.getGlobalTensorClass)().prototype.localResponseNormalization=function(e,t,r,n){return this.throwIfDisposed(),(0,rZ.localResponseNormalization)(this,e,t,r,n)};var rJ=e.i(24027);(0,rm.getGlobalTensorClass)().prototype.logSigmoid=function(){return this.throwIfDisposed(),(0,rJ.logSigmoid)(this)};var rQ=e.i(50472);(0,rm.getGlobalTensorClass)().prototype.logSoftmax=function(e){return this.throwIfDisposed(),(0,rQ.logSoftmax)(this,e)};var r0=e.i(86520);(0,rm.getGlobalTensorClass)().prototype.logSumExp=function(e,t){return this.throwIfDisposed(),(0,r0.logSumExp)(this,e,t)},(0,rm.getGlobalTensorClass)().prototype.log=function(){return this.throwIfDisposed(),(0,tI.log)(this)};var r1=e.i(22982);(0,rm.getGlobalTensorClass)().prototype.log1p=function(){return this.throwIfDisposed(),(0,r1.log1p)(this)},(0,rm.getGlobalTensorClass)().prototype.logicalAnd=function(e){return this.throwIfDisposed(),(0,el.logicalAnd)(this,e)},(0,rm.getGlobalTensorClass)().prototype.logicalNot=function(){return this.throwIfDisposed(),(0,tW.logicalNot)(this)};var r2=e.i(48312);(0,rm.getGlobalTensorClass)().prototype.logicalOr=function(e){return this.throwIfDisposed(),(0,r2.logicalOr)(this,e)};var r3=e.i(68624);(0,rm.getGlobalTensorClass)().prototype.logicalXor=function(e){return this.throwIfDisposed(),(0,r3.logicalXor)(this,e)},(0,rm.getGlobalTensorClass)().prototype.matMul=function(e,t,r){return this.throwIfDisposed(),(0,Q.matMul)(this,e,t,r)};var r4=e.i(94842);(0,rm.getGlobalTensorClass)().prototype.maxPool=function(e,t,r,n){return this.throwIfDisposed(),(0,r4.maxPool)(this,e,t,r,n)};var r5=e.i(76273);(0,rm.getGlobalTensorClass)().prototype.max=function(e,t){return this.throwIfDisposed(),(0,r5.max)(this,e,t)},(0,rm.getGlobalTensorClass)().prototype.maximum=function(e){return this.throwIfDisposed(),(0,rh.maximum)(this,e)};var r6=e.i(39401);(0,rm.getGlobalTensorClass)().prototype.mean=function(e,t){return this.throwIfDisposed(),(0,r6.mean)(this,e,t)};var r8=e.i(15988);(0,rm.getGlobalTensorClass)().prototype.min=function(e,t){return this.throwIfDisposed(),(0,r8.min)(this,e,t)};var r7=e.i(58651);(0,rm.getGlobalTensorClass)().prototype.minimum=function(e){return this.throwIfDisposed(),(0,r7.minimum)(this,e)};var r9=e.i(57554);(0,rm.getGlobalTensorClass)().prototype.mirrorPad=function(e,t){return this.throwIfDisposed(),(0,r9.mirrorPad)(this,e,t)};var ne=e.i(39872);(0,rm.getGlobalTensorClass)().prototype.mod=function(e){return this.throwIfDisposed(),(0,ne.mod)(this,e)},(0,rm.getGlobalTensorClass)().prototype.mul=function(e){return this.throwIfDisposed(),(0,b.mul)(this,e)},(0,rm.getGlobalTensorClass)().prototype.neg=function(){return this.throwIfDisposed(),(0,I.neg)(this)};var nt=e.i(69966);(0,rm.getGlobalTensorClass)().prototype.norm=function(e,t,r){return this.throwIfDisposed(),(0,nt.norm)(this,e,t,r)};var nr=e.i(5515);(0,rm.getGlobalTensorClass)().prototype.notEqual=function(e){return this.throwIfDisposed(),(0,nr.notEqual)(this,e)};var nn=e.i(97051);(0,rm.getGlobalTensorClass)().prototype.oneHot=function(e,t=1,r=0){return this.throwIfDisposed(),(0,nn.oneHot)(this,e,t,r)};var na=e.i(92047);(0,rm.getGlobalTensorClass)().prototype.onesLike=function(){return this.throwIfDisposed(),(0,na.onesLike)(this)},(0,rm.getGlobalTensorClass)().prototype.pad=function(e,t){return this.throwIfDisposed(),(0,tZ.pad)(this,e,t)};var ns=e.i(57901);(0,rm.getGlobalTensorClass)().prototype.pool=function(e,t,r,n,a,s){return this.throwIfDisposed(),(0,ns.pool)(this,e,t,r,n,a,s)},(0,rm.getGlobalTensorClass)().prototype.pow=function(e){return this.throwIfDisposed(),(0,tk.pow)(this,e)};var ni=e.i(52393);(0,rm.getGlobalTensorClass)().prototype.prelu=function(e){return this.throwIfDisposed(),(0,ni.prelu)(this,e)};var no=e.i(82772);(0,rm.getGlobalTensorClass)().prototype.prod=function(e,t){return this.throwIfDisposed(),(0,no.prod)(this,e,t)};var nl=e.i(53673);(0,rm.getGlobalTensorClass)().prototype.reciprocal=function(){return this.throwIfDisposed(),(0,nl.reciprocal)(this)};var nu=e.i(92462);(0,rm.getGlobalTensorClass)().prototype.relu=function(){return this.throwIfDisposed(),(0,nu.relu)(this)};var nh=e.i(77318);(0,rm.getGlobalTensorClass)().prototype.relu6=function(){return this.throwIfDisposed(),(0,nh.relu6)(this)},(0,rm.getGlobalTensorClass)().prototype.reshapeAs=function(e){return this.throwIfDisposed(),(0,E.reshape)(this,e.shape)},(0,rm.getGlobalTensorClass)().prototype.reshape=function(e){return this.throwIfDisposed(),(0,E.reshape)(this,e)};var nd=e.i(13747);(0,rm.getGlobalTensorClass)().prototype.resizeBilinear=function(e,t,r){return this.throwIfDisposed(),(0,nd.resizeBilinear)(this,e,t,r)};var np=e.i(94765);(0,rm.getGlobalTensorClass)().prototype.resizeNearestNeighbor=function(e,t,r){return this.throwIfDisposed(),(0,np.resizeNearestNeighbor)(this,e,t,r)},(0,rm.getGlobalTensorClass)().prototype.reverse=function(e){return this.throwIfDisposed(),(0,t_.reverse)(this,e)};var nc=e.i(73107);(0,rm.getGlobalTensorClass)().prototype.rfft=function(){return this.throwIfDisposed(),(0,nc.rfft)(this)};var nf=e.i(24642);(0,rm.getGlobalTensorClass)().prototype.round=function(){return this.throwIfDisposed(),(0,nf.round)(this)},(0,rm.getGlobalTensorClass)().prototype.rsqrt=function(){return this.throwIfDisposed(),(0,eG.rsqrt)(this)};var nm=e.i(42052);(0,rm.getGlobalTensorClass)().prototype.selu=function(){return this.throwIfDisposed(),(0,nm.selu)(this)};var ng=e.i(88791);(0,rm.getGlobalTensorClass)().prototype.separableConv2d=function(e,t,r,n,a,s){return this.throwIfDisposed(),(0,ng.separableConv2d)(this,e,t,r,n,a,s)},(0,rm.getGlobalTensorClass)().prototype.sigmoid=function(){return this.throwIfDisposed(),(0,t1.sigmoid)(this)};var nx=e.i(23497);(0,rm.getGlobalTensorClass)().prototype.sign=function(){return this.throwIfDisposed(),(0,nx.sign)(this)},(0,rm.getGlobalTensorClass)().prototype.sin=function(){return this.throwIfDisposed(),(0,eC.sin)(this)},(0,rm.getGlobalTensorClass)().prototype.sinh=function(){return this.throwIfDisposed(),(0,ek.sinh)(this)},(0,rm.getGlobalTensorClass)().prototype.slice=function(e,t){return this.throwIfDisposed(),(0,td.slice)(this,e,t)};var ny=e.i(75896);(0,rm.getGlobalTensorClass)().prototype.softmax=function(e){return this.throwIfDisposed(),(0,ny.softmax)(this,e)};var nb=e.i(76473);(0,rm.getGlobalTensorClass)().prototype.softplus=function(){return this.throwIfDisposed(),(0,nb.softplus)(this)},(0,rm.getGlobalTensorClass)().prototype.spaceToBatchND=function(e,t){return this.throwIfDisposed(),(0,et.spaceToBatchND)(this,e,t)},(0,rm.getGlobalTensorClass)().prototype.split=function(e,t){return this.throwIfDisposed(),(0,ep.split)(this,e,t)},(0,rm.getGlobalTensorClass)().prototype.sqrt=function(){return this.throwIfDisposed(),(0,S.sqrt)(this)},(0,rm.getGlobalTensorClass)().prototype.square=function(){return this.throwIfDisposed(),(0,T.square)(this)};var nv=e.i(20096);(0,rm.getGlobalTensorClass)().prototype.squaredDifference=function(e){return this.throwIfDisposed(),(0,nv.squaredDifference)(this,e)};var nw=e.i(43543);(0,rm.getGlobalTensorClass)().prototype.squeeze=function(e){return this.throwIfDisposed(),(0,nw.squeeze)(this,e)},(0,rm.getGlobalTensorClass)().prototype.stack=function(e,t){this.throwIfDisposed();let r=e instanceof rm.Tensor?[this,e]:[this,...e];return(0,eH.stack)(r,t)},(0,rm.getGlobalTensorClass)().prototype.step=function(e){return this.throwIfDisposed(),(0,v.step)(this,e)};var nC=e.i(96706);(0,rm.getGlobalTensorClass)().prototype.stridedSlice=function(e,t,r,n,a,s,i,o){return this.throwIfDisposed(),(0,nC.stridedSlice)(this,e,t,r,n,a,s,i,o)},(0,rm.getGlobalTensorClass)().prototype.sub=function(e){return this.throwIfDisposed(),(0,N.sub)(this,e)},(0,rm.getGlobalTensorClass)().prototype.sum=function(e,t){return this.throwIfDisposed(),(0,F.sum)(this,e,t)};var nI=e.i(64849);(0,rm.getGlobalTensorClass)().prototype.tan=function(){return this.throwIfDisposed(),(0,nI.tan)(this)};var nk=e.i(13195);(0,rm.getGlobalTensorClass)().prototype.tanh=function(){return this.throwIfDisposed(),(0,nk.tanh)(this)},(0,rm.getGlobalTensorClass)().prototype.tile=function(e){return this.throwIfDisposed(),(0,eU.tile)(this,e)},(0,rm.getGlobalTensorClass)().prototype.toBool=function(){return this.throwIfDisposed(),(0,y.cast)(this,"bool")},(0,rm.getGlobalTensorClass)().prototype.toFloat=function(){return this.throwIfDisposed(),(0,y.cast)(this,"float32")},(0,rm.getGlobalTensorClass)().prototype.toInt=function(){return this.throwIfDisposed(),(0,y.cast)(this,"int32")};var nS=e.i(88393);(0,rm.getGlobalTensorClass)().prototype.topk=function(e,t){return this.throwIfDisposed(),(0,nS.topk)(this,e,t)},(0,rm.getGlobalTensorClass)().prototype.transpose=function(e){return this.throwIfDisposed(),(0,e$.transpose)(this,e)};var nT=e.i(26806);(0,rm.getGlobalTensorClass)().prototype.unique=function(e){return this.throwIfDisposed(),(0,nT.unique)(this,e)},(0,rm.getGlobalTensorClass)().prototype.unsortedSegmentSum=function(e,t){return this.throwIfDisposed(),(0,eq.unsortedSegmentSum)(this,e,t)},(0,rm.getGlobalTensorClass)().prototype.unstack=function(e){return this.throwIfDisposed(),(0,tv.unstack)(this,e)},(0,rm.getGlobalTensorClass)().prototype.where=function(e,t){return this.throwIfDisposed(),(0,eu.where)(e,this,t)},(0,rm.getGlobalTensorClass)().prototype.zerosLike=function(){return this.throwIfDisposed(),(0,L.zerosLike)(this)};var nN=e.i(93599),n$=e.i(15950),nR=e.i(95159);class nA extends Error{constructor(e){super(e),Object.setPrototypeOf(this,nA.prototype)}}class nE extends Error{constructor(e){super(e),Object.setPrototypeOf(this,nE.prototype)}}class nF extends Error{constructor(e){super(e),Object.setPrototypeOf(this,nF.prototype)}}class nD extends Error{constructor(e){super(e),Object.setPrototypeOf(this,nD.prototype)}}class nO extends Error{constructor(e){super(e),Object.setPrototypeOf(this,nO.prototype)}}class nL{constructor(e){this.maxEntries=e||100,this.cache=new Map}get(e){let t;return this.cache.has(e)&&(t=this.cache.get(e),this.cache.delete(e),this.cache.set(e,t)),t}put(e,t){if(this.cache.has(e))this.cache.delete(e);else if(this.cache.size>=this.maxEntries){let e=this.cache.keys().next().value;this.cache.delete(e)}this.cache.set(e,t)}getMaxEntries(){return this.maxEntries}setMaxEntries(e){if(e<0)throw Error(`The maxEntries of LRU caches must be at least 0, but got ${e}.`);if(this.maxEntries>e)for(let t=0;t<this.maxEntries-e;t++){let e=this.cache.keys().next().value;this.cache.delete(e)}this.maxEntries=e}}function nz(e,t){if(Array.isArray(e)){let r=[];for(let n=0;n<t;n++)r=r.concat(e);return r}{let r=Array(t);return r.fill(e),r}}function n_(e,t){if(!e)throw new nO(t)}function nM(e,t){let r=0;for(let n of e)n===t&&r++;return r}function nP(e){return 1===e.length?e[0]:e}function nB(e){return Array.isArray(e)?e:[e]}function nW(e){let t=e.replace(/(.)([A-Z][a-z0-9]+)/g,"$1_$2").replace(/([a-z])([A-Z])/g,"$1_$2").toLowerCase();return"_"!==t[0]?t:"private"+t}function nG(e){return e.length<=1||-1===e.indexOf("_")?e:e.replace(/[_]+(\w|$)/g,(e,t)=>t.toUpperCase())}let nU={};function nV(e){if(null==e)return null;let t={};return t.className=e.getClassName(),t.config=e.getConfig(),t}function nH(e,t={},r={},n="object",a=!1){if("string"==typeof e){let a;if(e in r)a=r[e];else if(e in nU)a=nU[e];else if(null==(a=t[e]))throw new nF(`Unknown ${n}: ${e}. This may be due to one of the following reasons:
1. The ${n} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${n} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);return a}{let s,i;if(null==e.className||null==e.config)throw new nF(`${n}: Improper config format: ${JSON.stringify(e)}.
'className' and 'config' must set.`);let o=e.className;if(o in r?[s,i]=r[o]:o in nU?[s,i]=nU.className:o in t&&([s,i]=t[o]),null==s)throw new nF(`Unknown ${n}: ${o}. This may be due to one of the following reasons:
1. The ${n} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${n} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);if(null!=i){let t={};for(let e of Object.keys(nU))t[e]=nU[e];for(let e of Object.keys(r))t[e]=r[e];e.config.customObjects=t;let n=Object.assign({},nU);for(let e of Object.keys(r))nU[e]=r[e];!function e(t){if(null!=t&&"object"==typeof t)if(Array.isArray(t))t.forEach(t=>e(t));else for(let r of Object.keys(t)){let n=t[r];null!=n&&"object"==typeof n&&(Array.isArray(n)||"ndarray"!==n.type||"number"!=typeof n.value?e(n):t[r]=n.value)}}(e.config);let o=i(s,e.config,r,a);return nU=Object.assign({},n),o}{let t=Object.assign({},nU);for(let e of Object.keys(r))nU[e]=r[e];let n=new s(e.config);return nU=Object.assign({},t),n}}}function nq(e,t){return -1*(e<t?-1:+(e>t))}function nj(e){if(null==e)return e;let t=[];for(let r of e)-1===t.indexOf(r)&&t.push(r);return t}function nX(e,t,r){if(null!=r&&0>e.indexOf(r))throw new nF(`${r} is not a valid ${t}.  Valid values are ${e} or null/undefined.`)}function nK(e,t,r=0,n=1/0){return n_(r>=0),n_(n>=r),Array.isArray(e)&&e.length>=r&&e.length<=n&&e.every(e=>typeof e===t)}function nY(e,t){Array.isArray(e)?(nR.util.assert(e.length>0,()=>`${t} is unexpectedly an empty array.`),e.forEach((e,r)=>nY(e,`element ${r+1} of ${t}`))):nR.util.assert(Number.isInteger(e)&&e>0,()=>`Expected ${t} to be a positive integer, but got ${function e(t){return null===t?"null":Array.isArray(t)?"["+t.map(t=>e(t)).join(",")+"]":"string"==typeof t?`"${t}"`:`${t}`}(e)}.`)}function nZ(e){return"relu"===e?"relu":"linear"===e?"linear":"elu"===e?"elu":null}var nJ=e.i(86907),nJ=nJ;let nQ=0,n0={};function n1(e=""){return e in n0||(n0[e]=0),n0[e]+=1,e+n0[e].toString()}var nJ=nJ;let n2=["channelsFirst","channelsLast"],n3=["nearest","bilinear"],n4=["valid","same","causal"],n5=["max","avg"],n6=["sum","mul","concat","ave"],n8=new Map;function n7(e){nX(n2,"DataFormat",e)}function n9(e){nX(n4,"PaddingMode",e)}function ae(e){nX(n5,"PoolMode",e)}let at=[];function ar(e,t){at.push(e);try{let e=t();return at.pop(),e}catch(e){throw at.pop(),e}}function an(e){if(!ai(e))throw Error("Not a valid tensor name: '"+e+"'");return(0===at.length?"":at.join("/")+"/")+e}function aa(e){if(!ai(e))throw Error("Not a valid tensor name: '"+e+"'");n8.has(e)||n8.set(e,0);let t=n8.get(e);if(n8.set(e,n8.get(e)+1),!(t>0))return e;{let r=`${e}_${t}`;return n8.set(r,1),r}}let as=new RegExp(/^[A-Za-z0-9][-A-Za-z0-9\._\/]*$/);function ai(e){return!!e.match(as)}var ao=e.i(17132),al=e.i(82068),au=e.i(56543),nJ=nJ,ah=e.i(77913),ad=e.i(50611),ap=e.i(95657),ac=e.i(35666),af=e.i(77276),am=e.i(46291),ag=e.i(1243),ax=e.i(77444),ay=e.i(9109),ab=e.i(8656),av=e.i(72693),av=av,aw=e.i(27445),aC=e.i(37195);function aI(e,t,r){null==t&&(t=0),null==r&&(r=e.length);let n=1;for(let a=t;a<r;++a)n*=e[a];return n}function ak(e){if(0===e.length)return NaN;let t=1/0;for(let r=0;r<e.length;r++){let n=e[r];n<t&&(t=n)}return t}function aS(e){if(0===e.length)return NaN;let t=-1/0;for(let r=0;r<e.length;r++){let n=e[r];n>t&&(t=n)}return t}function aT(e,t){if(t<e)throw new nF(`end (${t}) < begin (${e}) is forbidden.`);let r=[];for(let n=e;n<t;++n)r.push(n);return r}function aN(){return null==t&&(t=(0,n$.backend)().epsilon()),t}function a$(){return"channelsLast"}function aR(e,t){return y.cast(e,t)}function aA(e,t=-1){let r=e.shape.slice();return t<0&&(t=r.length+t+1),r.splice(t,0,1),E.reshape(e,r)}function aE(e,t,r){return(0,n$.tidy)(()=>{switch(e.rank){case 1:return ad.slice1d(e,t,r);case 2:return ap.slice2d(e,[t,0],[r,e.shape[1]]);case 3:return ac.slice3d(e,[t,0,0],[r,e.shape[1],e.shape[2]]);case 4:return af.slice4d(e,[t,0,0,0],[r,e.shape[1],e.shape[2],e.shape[3]]);case 5:return td.slice(e,[t,0,0,0,0],[r,e.shape[1],e.shape[2],e.shape[3],e.shape[4]]);case 6:return td.slice(e,[t,0,0,0,0,0],[r,e.shape[1],e.shape[2],e.shape[3],e.shape[4],e.shape[5]]);default:throw new nF(`sliceAlongFirstAxis() received an unsupported tensor rank: ${e.rank}`)}})}function aF(e,t,r){return(0,n$.tidy)(()=>{switch(e.rank){case 1:return ad.slice1d(e,t,r);case 2:return ap.slice2d(e,[0,t],[e.shape[0],r]);case 3:return ac.slice3d(e,[0,0,t],[e.shape[0],e.shape[1],r]);case 4:return af.slice4d(e,[0,0,0,t],[e.shape[0],e.shape[1],e.shape[2],r]);default:throw new nF(`sliceAlongLastAxis() received an unsupported tensor rank: ${e.rank}`)}})}function aD(e,t,r,n){return(0,n$.tidy)(()=>{switch(e.rank){case 1:return ad.slice1d(e,t,r);case 2:switch(n){case 1:return aE(e,t,r);case 2:return aF(e,t,r);default:throw new nF(`The axis is not within the rank of the tensor ${n}`)}case 3:switch(n){case 1:return aE(e,t,r);case 2:return ac.slice3d(e,[0,t,0],[e.shape[0],r,e.shape[2]]);case 3:return aF(e,t,r);default:throw new nF(`The axis is not within the rank of the tensor ${n}`)}case 4:switch(n){case 1:return aE(e,t,r);case 2:return af.slice4d(e,[0,t,0,0],[e.shape[0],r,e.shape[2],e.shape[3]]);case 3:return af.slice4d(e,[0,0,t,0],[e.shape[0],e.shape[1],r,e.shape[3]]);case 4:return aF(e,t,r);default:throw new nF(`The axis is not within the rank of the tensor ${n}`)}default:throw new nF(`sliceAlongLastAxis() received an unsupported tensor rank: ${e.rank}`)}})}function aO(e,t=-1){let r;return t<0&&(t=0!==(r=e[0].rank)?r:0),t===e[0].rank&&(t=-1),t5.concat(e,t)}function aL(e,t){switch(e.rank){case 1:return am.concat1d([e,t]);case 2:return ag.concat2d([e,t],0);case 3:return ax.concat3d([e,t],0);case 4:return ay.concat4d([e,t],0);default:throw new nF(`concatAlongFirstAxis() received an unsupported tensor rank: ${e.rank}`)}}function az(e,t){if(Array.isArray(t)||(t=[t]),e.rank!==t.length)throw new nF(`The length of input n (${t.length}) does not match the number of dimensions in input x (${e.rank})`);return eU.tile(e,t)}function a_(e,t=0,r=1,n,a){return ab.randomNormal(e,t,r,n,a)}function aM(e,t,r,n){if(e.rank<2||t.rank<2)throw new nD(`dot requires both inputs to be rank >= 2 but got x shape = ${e.shape} and y shape = ${t.shape}`);if(t.rank>=3&&e.shape.slice(-1)[0]!==t.shape.slice(-2)[0])throw new nD(`If rank y >= 3, then the second last dim of y must equal the last dim of x but got x shape = ${e.shape} and  y shape = ${t.shape}`);if(2===e.rank&&2===t.rank)return av.matMul({a:e,b:t,transposeA:!1,transposeB:!1,bias:n?aW(e.rank,n,a$()):null,activation:r});{let a=e.shape.slice(),s=a.pop();e=E.reshape(e,[-1,s]);let i=t.shape.slice(),o=i.pop(),l=i.pop(),u=[...i,o],h=Array.from({length:t.rank},(e,r)=>0===r?t.rank-2:r<=t.rank-2?r-1:r);t=E.reshape(e$.transpose(t,h),[l,-1]);let d=[...a,...u];return E.reshape(av.matMul({a:e,b:t,transposeA:!1,transposeB:!1,bias:n?aW(e.rank,n,a$()):null,activation:r}),d)}}function aP(e,t,r){return(0,n$.tidy)(()=>(t=Array.isArray(t)?(0,aC.tensor1d)(t,"int32"):y.cast(t,"int32"),ru.gather(e,t,r)))}function aB(e){return b.mul(e,e)}function aW(e,t,r){let n=t.shape;if(1!==t.rank&&t.rank!==e)throw new nF(`Unexpected bias dimensions: ${t.rank}; expected it to be 1 or ${e}`);if(5===e){if("channelsFirst"===r)if(1===n.length)return E.reshape(t,[1,n[0],1,1,1]);else return E.reshape(t,[1,n[3],n[0],n[1],n[2]]);else if("channelsLast"===r)if(1===n.length)return E.reshape(t,[1,1,1,1,n[0]]);else return E.reshape(t,[1].concat(n))}else if(4===e){if("channelsFirst"===r)if(1===n.length)return E.reshape(t,[1,n[0],1,1]);else return E.reshape(t,[1,n[2],n[0],n[1]]);else if("channelsLast"===r)if(1===n.length)return E.reshape(t,[1,1,1,n[0]]);else return E.reshape(t,[1].concat(n))}else if(3===e){if("channelsFirst"===r)if(1===n.length)return E.reshape(t,[1,n[0],1]);else return E.reshape(t,[1,n[1],n[0]]);else if("channelsLast"===r)if(1===n.length)return E.reshape(t,[1,1,n[0]]);else return E.reshape(t,[1].concat(n))}else if(e<3)return t;throw new nF(`Unsupported input rank by biasAdd: ${t.rank}`)}function aG(e,t,r){return(0,n$.tidy)(()=>(null==r&&(r=a$()),n7(r),P.add(e,aW(e.rank,t,r))))}function aU(e,t,r,n){return(0,n$.tidy)(()=>aw.dropout(e,t,r,n))}function aV(e,t,r=!1){return r?e():t()}let aH=["fanIn","fanOut","fanAvg"],aq=["normal","uniform","truncatedNormal"];class aj extends nJ.Serializable{fromConfigUsesCustomObjects(){return!1}getConfig(){return{}}}class aX extends aj{apply(e,t){return(0,tx.zeros)(e,t)}}aX.className="Zeros",nJ.registerClass(aX);class aK extends aj{apply(e,t){return(0,to.ones)(e,t)}}aK.className="Ones",nJ.registerClass(aK);class aY extends aj{constructor(e){if(super(),"object"!=typeof e)throw new nF(`Expected argument of type ConstantConfig but got ${e}`);if(void 0===e.value)throw new nF(`config must have value set but got ${e}`);this.value=e.value}apply(e,t){return(0,n$.tidy)(()=>(0,b.mul)((0,k.scalar)(this.value),(0,to.ones)(e,t)))}getConfig(){return{value:this.value}}}aY.className="Constant",nJ.registerClass(aY);class aZ extends aj{constructor(e){super(),this.DEFAULT_MINVAL=-.05,this.DEFAULT_MAXVAL=.05,this.minval=e.minval||this.DEFAULT_MINVAL,this.maxval=e.maxval||this.DEFAULT_MAXVAL,this.seed=e.seed}apply(e,t){return(0,au.randomUniform)(e,this.minval,this.maxval,t,this.seed)}getConfig(){return{minval:this.minval,maxval:this.maxval,seed:this.seed}}}aZ.className="RandomUniform",nJ.registerClass(aZ);class aJ extends aj{constructor(e){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=e.mean||this.DEFAULT_MEAN,this.stddev=e.stddev||this.DEFAULT_STDDEV,this.seed=e.seed}apply(e,t){if("float32"!==(t=t||"float32")&&"int32"!==t)throw new nD(`randomNormal does not support dType ${t}.`);return a_(e,this.mean,this.stddev,t,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}aJ.className="RandomNormal",nJ.registerClass(aJ);class aQ extends aj{constructor(e){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=e.mean||this.DEFAULT_MEAN,this.stddev=e.stddev||this.DEFAULT_STDDEV,this.seed=e.seed}apply(e,t){if("float32"!==(t=t||"float32")&&"int32"!==t)throw new nD(`truncatedNormal does not support dType ${t}.`);return(0,ah.truncatedNormal)(e,this.mean,this.stddev,t,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}aQ.className="TruncatedNormal",nJ.registerClass(aQ);class a0 extends aj{constructor(e){super(),this.gain=null!=e.gain?e.gain:1}apply(e,t){return(0,n$.tidy)(()=>{if(2===e.length&&e[0]===e[1])return(0,b.mul)(this.gain,(0,ao.eye)(e[0]));throw new nF("Identity matrix initializer can only be used for 2D square matrices.")})}getConfig(){return{gain:this.gain}}}a0.className="Identity",nJ.registerClass(a0);class a1 extends aj{constructor(e){if(super(),e.scale<0)throw new nF(`scale must be a positive float. Got: ${e.scale}`);this.scale=null==e.scale?1:e.scale,this.mode=null==e.mode?"fanIn":e.mode,function(e){nX(aH,"FanMode",e)}(this.mode),this.distribution=null==e.distribution?"normal":e.distribution,function(e){nX(aq,"Distribution",e)}(this.distribution),this.seed=e.seed}apply(e,t){let r=function(e,t="channelsLast"){let r,n;if(n7(t),2===e.length)r=e[0],n=e[1];else if(-1!==[3,4,5].indexOf(e.length)){if("channelsFirst"===t){let t=aI(e,2);r=e[1]*t,n=e[0]*t}else if("channelsLast"===t){let t=aI(e,0,e.length-2);r=e[e.length-2]*t,n=e[e.length-1]*t}}else{let t=aI(e);r=Math.sqrt(t),n=Math.sqrt(t)}return[r,n]}(e),n=r[0],a=r[1],s=this.scale;if("fanIn"===this.mode?s/=Math.max(1,n):"fanOut"===this.mode?s/=Math.max(1,a):s/=Math.max(1,(n+a)/2),"normal"===this.distribution){let r=Math.sqrt(s);if("float32"!==(t=t||"float32")&&"int32"!==t)throw new nD(`${this.getClassName()} does not support dType ${t}.`);return(0,ah.truncatedNormal)(e,0,r,t,this.seed)}{let r=Math.sqrt(3*s);return(0,au.randomUniform)(e,-r,r,t,this.seed)}}getConfig(){return{scale:this.scale,mode:this.mode,distribution:this.distribution,seed:this.seed}}}a1.className="VarianceScaling",nJ.registerClass(a1);class a2 extends a1{constructor(e){super({scale:1,mode:"fanAvg",distribution:"uniform",seed:null==e?null:e.seed})}getClassName(){return a1.className}}a2.className="GlorotUniform",nJ.registerClass(a2);class a3 extends a1{constructor(e){super({scale:1,mode:"fanAvg",distribution:"normal",seed:null==e?null:e.seed})}getClassName(){return a1.className}}a3.className="GlorotNormal",nJ.registerClass(a3);class a4 extends a1{constructor(e){super({scale:2,mode:"fanIn",distribution:"normal",seed:null==e?null:e.seed})}getClassName(){return a1.className}}a4.className="HeNormal",nJ.registerClass(a4);class a5 extends a1{constructor(e){super({scale:2,mode:"fanIn",distribution:"uniform",seed:null==e?null:e.seed})}getClassName(){return a1.className}}a5.className="HeUniform",nJ.registerClass(a5);class a6 extends a1{constructor(e){super({scale:1,mode:"fanIn",distribution:"normal",seed:null==e?null:e.seed})}getClassName(){return a1.className}}a6.className="LeCunNormal",nJ.registerClass(a6);class a8 extends a1{constructor(e){super({scale:1,mode:"fanIn",distribution:"uniform",seed:null==e?null:e.seed})}getClassName(){return a1.className}}a8.className="LeCunUniform",nJ.registerClass(a8);class a7 extends aj{constructor(e){super(),this.DEFAULT_GAIN=1,this.ELEMENTS_WARN_SLOW=2e3,this.gain=null==e.gain?this.DEFAULT_GAIN:e.gain,this.seed=e.seed}apply(e,t){return(0,n$.tidy)(()=>{if(e.length<2)throw new nD("Shape must be at least 2D.");if("int32"!==t&&"float32"!==t&&void 0!==t)throw TypeError(`Unsupported data type ${t}.`);let r=nR.util.sizeFromShape(e.slice(0,-1)),n=e[e.length-1],a=r*n;a>this.ELEMENTS_WARN_SLOW&&console.warn(`Orthogonal initializer is being called on a matrix with more than ${this.ELEMENTS_WARN_SLOW} (${a}) elements: Slowness may result.`);let s=a_([Math.max(n,r),Math.min(n,r)],0,1,t,this.seed),i=al.linalg.qr(s,!1),o=i[0],l=i[1].flatten().stridedSlice([0],[Math.min(n,r)*Math.min(n,r)],[Math.min(n,r)+1]);return o=(0,b.mul)(o,l.sign()),r<n&&(o=o.transpose()),(0,b.mul)((0,k.scalar)(this.gain),o.reshape(e))})}getConfig(){return{gain:this.gain,seed:this.seed}}}a7.className="Orthogonal",nJ.registerClass(a7);let a9={constant:"Constant",glorotNormal:"GlorotNormal",glorotUniform:"GlorotUniform",heNormal:"HeNormal",heUniform:"HeUniform",identity:"Identity",leCunNormal:"LeCunNormal",leCunUniform:"LeCunUniform",ones:"Ones",orthogonal:"Orthogonal",randomNormal:"RandomNormal",randomUniform:"RandomUniform",truncatedNormal:"TruncatedNormal",varianceScaling:"VarianceScaling",zeros:"Zeros"};function se(e,t={}){return nH(e,nJ.SerializationMap.getMap().classNameMap,t,"initializer")}function st(e){if("string"==typeof e){let t=e in a9?a9[e]:e;if("GlorotNormal"===t)return new a3;{if("GlorotUniform"===t)return new a2;if("HeNormal"===t)return new a4;if("HeUniform"===t)return new a5;if("LeCunNormal"===t)return new a6;if("LeCunUniform"===t)return new a8;let e={};return e.className=t,e.config={},se(e)}}return e instanceof aj?e:se(e)}function sr(e){return Array.isArray(e)&&Array.isArray(e[0])}function sn(e){return 0===e.length?[]:Array.isArray(e[0])?e:[e]}function sa(e){let t;if(Array.isArray(e)){if(1!==e.length)throw new nF(`Expected Tensor length to be 1; got ${e.length}`);t=e[0]}else t=e;return t}function ss(e){if(!(Array.isArray(e)&&Array.isArray(e[0])))return e;if(1===e.length)return e[0];throw new nF(`Expected exactly 1 Shape; got ${e.length}`)}function si(e){let t=0;for(let r of e)0===r.shape.length?t+=1:t+=r.shape.reduce((e,t)=>e*t);return t}var so=e.i(79588);let sl="Variable";class su{constructor(e,t="float32",r=sl,n=!0,a=null){this.dtype=null==t?"float32":t,this.shape=e.shape,this.id=nQ++,r=null==r?sl:r,this.originalName=an(r),this.name=aa(this.originalName),this.trainable_=n,this.constraint=a,this.val=so.variable(e,this.trainable_,this.name,this.dtype)}read(){return this.assertNotDisposed(),this.val}write(e){return this.assertNotDisposed(),function(e,t){if(e.shape.toString()!==t.shape.toString())throw Error("Shape mismatch: "+JSON.stringify(e.shape)+" vs. "+JSON.stringify(t.shape))}(this.val,e),this.val.id!==e.id&&(this.val.assign(e),null!=this.constraint&&this.val.assign(this.constraint.apply(this.val))),this}dispose(){this.assertNotDisposed(),this.val.dispose()}assertNotDisposed(){if(this.val.isDisposed)throw Error(`LayersVariable ${this.name} is already disposed.`)}get trainable(){return this.trainable_}set trainable(e){this.trainable_=e,this.val.trainable=e}}function sh(e){return e.map(e=>e.read())}function sd(e){e.forEach(e=>{e[0].write(e[1])})}class sp{constructor(e){this.dtype=e.dtype,this.shape=e.shape,null!=e.shape?this.ndim=e.shape.length:this.ndim=e.ndim,this.maxNDim=e.maxNDim,this.minNDim=e.minNDim,this.axes=e.axes||{}}}class sc{constructor(e,t,r,n,a,s,i){this.dtype=e,this.shape=t,this.sourceLayer=r,this.inputs=n,this.callArgs=a,this.outputTensorIndex=i,this.id=nQ++,null!=s&&(this.originalName=an(s),this.name=aa(this.originalName)),this.rank=t.length}}let sf=0;class sm{constructor(e,t){for(const r of(this.callArgs=t,this.id=sf++,this.outboundLayer=e.outboundLayer,this.inboundLayers=e.inboundLayers,this.nodeIndices=e.nodeIndices,this.tensorIndices=e.tensorIndices,this.inputTensors=e.inputTensors,this.outputTensors=e.outputTensors,this.inputMasks=e.inputMasks,this.outputMasks=e.outputMasks,this.inputShapes=e.inputShapes,this.outputShapes=e.outputShapes,e.inboundLayers))null!=r&&r.outboundNodes.push(this);e.outboundLayer.inboundNodes.push(this)}getConfig(){let e=[];for(let t of this.inboundLayers)null!=t?e.push(t.name):e.push(null);return{outboundLayer:this.outboundLayer?this.outboundLayer.name:null,inboundLayers:e,nodeIndices:this.nodeIndices,tensorIndices:this.tensorIndices}}}let sg=0;class sx extends nJ.Serializable{constructor(e={}){super(),this._callHook=null,this._addedWeightNames=[],this._stateful=!1,this.id=sg++,this.activityRegularizer=null,this.inputSpec=null,this.supportsMasking=!1,this._trainableWeights=[],this._nonTrainableWeights=[],this._losses=[],this._updates=[],this._built=!1,this.inboundNodes=[],this.outboundNodes=[];let t=e.name;if(!t){const e=this.getClassName();t=nW(e)+"_"+n1(e)}if(this.name=t,this.trainable_=null==e.trainable||e.trainable,null!=e.inputShape||null!=e.batchInputShape){let t;if(null!=e.batchInputShape)t=e.batchInputShape;else if(null!=e.inputShape){let r=null;null!=e.batchSize&&(r=e.batchSize),t=[r].concat(e.inputShape)}this.batchInputShape=t;let r=e.dtype;null==r&&(r=e.inputDType),null==r&&(r="float32"),this.dtype=r}null!=e.weights?this.initialWeights=e.weights:this.initialWeights=null,this._refCount=null,this.fastWeightInitDuringBuild=!1}static nodeKey(e,t){return e.name+"_ib-"+t.toString()}getNodeAtIndex(e,t){if(0===this.inboundNodes.length)throw new nE(`The layer has never been called and thus has no defined ${t}.`);if(this.inboundNodes.length<=e)throw new nF(`Asked to get ${t} at node ${e}, but the layer has only ${this.inboundNodes.length} inbound nodes.`);return this.inboundNodes[e]}getInputAt(e){return nP(this.getNodeAtIndex(e,"input").inputTensors)}getOutputAt(e){return nP(this.getNodeAtIndex(e,"output").outputTensors)}get input(){if(this.inboundNodes.length>1)throw new nA(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer input" is ill-defined. Use \`getInputAt(nodeIndex)\` instead.`);if(0===this.inboundNodes.length)throw new nA(`Layer ${this.name} is not connected, no input to return.`);return nP(this.getNodeAtIndex(0,"input").inputTensors)}get output(){if(0===this.inboundNodes.length)throw new nA(`Layer ${this.name} has no inbound nodes.`);if(this.inboundNodes.length>1)throw new nA(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer output" is ill-defined. Use \`getOutputAt(nodeIndex)\` instead.`);return nP(this.getNodeAtIndex(0,"output").outputTensors)}get losses(){return this._losses}calculateLosses(){return this.losses.map(e=>e())}get updates(){return this._updates}get built(){return this._built}set built(e){this._built=e}get trainable(){return this.trainable_}set trainable(e){this._trainableWeights.forEach(t=>t.trainable=e),this.trainable_=e}get trainableWeights(){return this.trainable_?this._trainableWeights.filter(e=>e.trainable):[]}set trainableWeights(e){this._trainableWeights=e}get nonTrainableWeights(){return this.trainable?this._trainableWeights.filter(e=>!e.trainable).concat(this._nonTrainableWeights):this._trainableWeights.concat(this._nonTrainableWeights)}set nonTrainableWeights(e){this._nonTrainableWeights=e}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}get stateful(){return this._stateful}resetStates(){if(!this.stateful)throw Error("Cannot call the resetStates() method of a non-stateful Layer object.")}assertInputCompatibility(e){let t=nB(e);if(null==this.inputSpec||0===this.inputSpec.length)return;let r=nB(this.inputSpec);if(t.length!==r.length)throw new nF(`Layer ${this.name} expects ${r.length} inputs, but it received ${t.length} input tensors. Input received: ${e}`);for(let e=0;e<t.length;e++){let n=t[e],a=r[e];if(null==a)continue;let s=n.rank;if(null!=a.ndim&&s!==a.ndim)throw new nF(`Input ${e} is incompatible with layer ${this.name}: expected ndim=${a.ndim}, found ndim=${s}`);if(null!=a.maxNDim&&s>a.maxNDim)throw new nF(`Input ${e} is incompatible with layer ${this.name}: expected max_ndim=${a.maxNDim}, found ndim=${s}`);if(null!=a.minNDim&&s<a.minNDim)throw new nF(`Input ${e} is incompatible with layer ${this.name}: expected min_ndim=${a.minNDim}, found ndim=${s}.`);if(null!=a.dtype&&n.dtype!==a.dtype)throw new nF(`Input ${e} is incompatible with layer ${this.name} : expected dtype=${a.dtype}, found dtype=${n.dtype}.`);if(a.axes){let t=n.shape;for(let r in a.axes){let n=Number(r),s=a.axes[r],i=n>=0?t[n]:t[t.length+n];if(null!=s&&-1===[s,null].indexOf(i))throw new nF(`Input ${e} is incompatible with layer ${this.name}: expected axis ${n} of input shape to have value ${s} but got shape ${t}.`)}}if(null!=a.shape)for(let t=0;t<a.shape.length;++t){let r=a.shape[t],s=n.shape[t];if(null!=r&&null!=s&&r!==s)throw new nF(`Input ${e} is incompatible with layer ${this.name}: expected shape=${a.shape}, found shape=${n.shape}.`)}}}call(e,t){return e}invokeCallHook(e,t){null!=this._callHook&&this._callHook(e,t)}setCallHook(e){this._callHook=e}clearCallHook(){this._callHook=null}apply(e,t){t=t||{},this.assertNotDisposed();let r=nB(e),n=function(e){let t=!0;for(let r of nB(e))if(!(r instanceof sc)){t=!1;break}return t}(e),a=function(e){let t=!0;for(let r of nB(e))if(r instanceof sc){t=!1;break}return t}(e);if(n===a)throw new nF("Arguments to apply() must be all SymbolicTensors or all Tensors");return ar(this.name,()=>{if(!this.built){this.assertInputCompatibility(e);let t=[];for(let r of nB(e))t.push(r.shape);this.build(nP(t)),this.built=!0,this.initialWeights&&this.setWeights(this.initialWeights),null===this._refCount&&a&&(this._refCount=1)}if(this.assertInputCompatibility(e),a){let n=this.call(e,t);this.supportsMasking&&this.setMaskMetadata(e,n);let a=nB(n),s=[];for(let e of a)-1!==r.indexOf(e)&&(e=e.clone()),s.push(e);if(n=nP(s),null!=this.activityRegularizer)throw new nD("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return n}{let r,n=function(e){e=nB(e);let t=[];for(let r of e)t.push(r.shape);return nP(t)}(e),a=this.computeOutputShape(n),s="float32";if(this.warnOnIncompatibleInputShape(Array.isArray(e)?n[0]:n),r=null!=a&&a.length>0&&Array.isArray(a[0])?a.map((r,n)=>new sc(s,r,this,nB(e),t,this.name,n)):new sc(s,a,this,nB(e),t,this.name),this.addInboundNode(e,r,null,null,n,a,t),this._refCount++,null!=this.activityRegularizer)throw new nD("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return r}})}warnOnIncompatibleInputShape(e){if(null!=this.batchInputShape)if(e.length!==this.batchInputShape.length)console.warn(`The rank of the input tensor provided (shape: ${JSON.stringify(e)}) does not match that of the batchInputShape (${JSON.stringify(this.batchInputShape)}) of the layer ${this.name}`);else{let t=!1;this.batchInputShape.forEach((r,n)=>{null!=r&&null!=e[n]&&e[n]!==r&&(t=!0)}),t&&console.warn(`The shape of the input tensor (${JSON.stringify(e)}) does not match the expectation of layer ${this.name}: ${JSON.stringify(this.batchInputShape)}`)}}get outputShape(){if(null==this.inboundNodes||0===this.inboundNodes.length)throw new nA(`The layer ${this.name} has never been called and thus has no defined output shape.`);let e=[];for(let t of this.inboundNodes){let r=JSON.stringify(t.outputShapes);-1===e.indexOf(r)&&e.push(r)}if(1===e.length){let e=this.inboundNodes[0].outputShapes;return Array.isArray(e)&&Array.isArray(e[0])&&1===e.length?e[0]:e}throw new nA(`The layer ${this.name} has multiple inbound nodes with different output shapes. Hence the notion of "output shape" is ill-defined for the layer.`)}countParams(){if(!this.built)throw new nE(`You tried to call countParams() on ${this.name}, but the layer is not built yet. Build it first by calling build(batchInputShape).`);return si(this.weights)}build(e){this.built=!0}getWeights(e=!1){return sh(e?this.trainableWeights:this.weights)}setWeights(e){(0,n$.tidy)(()=>{let t=this.weights;if(t.length!==e.length)throw new nF(`You called setWeights(weights) on layer "${this.name}" with a weight list of length ${e.length}, but the layer was expecting ${t.length} weights. Provided weights: ${e}...`);if(0===t.length)return;let r=[],n=sh(t);for(let a=0;a<n.length;++a){let s=n[a],i=t[a],o=e[a];if(!nR.util.arraysEqual(s.shape,o.shape))throw new nF(`Layer weight shape ${s.shape} not compatible with provided weight shape ${o.shape}`);r.push([i,o])}sd(r)})}addWeight(e,t,r,n,a,s,i,o){if(-1!==this._addedWeightNames.indexOf(e))throw new nF(`Duplicate weight name ${e} for layer ${this.name}`);this._addedWeightNames.push(e),null==r&&(r="float32"),this.fastWeightInitDuringBuild&&(n=null!=o?o():st("zeros"));let l=n.apply(t,r),u=new su(l,r,e,s,i);return l.dispose(),null!=a&&this.addLoss(()=>a.apply(u.read())),null==s&&(s=!0),s?this._trainableWeights.push(u):this._nonTrainableWeights.push(u),u}setFastWeightInitDuringBuild(e){this.fastWeightInitDuringBuild=e}addLoss(e){null==e||Array.isArray(e)&&0===e.length||(e=nB(e),void 0!==this._losses&&null!==this._losses&&this.losses.push(...e))}computeOutputShape(e){return e}computeMask(e,t){if(!this.supportsMasking){if(null!=t)if(Array.isArray(t))t.forEach(e=>{if(null!=e)throw TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`)});else throw TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`);return null}return t}setMaskMetadata(e,t,r){if(!this.supportsMasking)return;let n=this.computeMask(e,r),a=nB(t),s=nB(n);if(a.length!==s.length)throw Error(`${this.name} outputs ${a.length} tensors but ${a.length} masks for those tensors`);for(let e=0;e<a.length;e++)a[e].kerasMask=s[e]}addInboundNode(e,t,r,n,a,s,i=null){let o=nB(e);t=nB(t),r=nB(r),n=nB(n),a=sn(a),s=sn(s);let l=[],u=[],h=[];for(let e of o)l.push(e.sourceLayer),u.push(e.nodeIndex),h.push(e.tensorIndex);new sm({outboundLayer:this,inboundLayers:l,nodeIndices:u,tensorIndices:h,inputTensors:o,outputTensors:t,inputMasks:r,outputMasks:n,inputShapes:a,outputShapes:s},i);for(let e=0;e<t.length;e++)t[e].sourceLayer=this,t[e].nodeIndex=this.inboundNodes.length-1,t[e].tensorIndex=e}getConfig(){let e={name:this.name,trainable:this.trainable};return null!=this.batchInputShape&&(e.batchInputShape=this.batchInputShape),null!=this.dtype&&(e.dtype=this.dtype),e}disposeWeights(){return this.weights.forEach(e=>e.dispose()),this.weights.length}assertNotDisposed(){if(0===this._refCount)throw Error(`Layer '${this.name}' is already disposed.`)}dispose(){if(!this.built)throw Error(`Cannot dispose Layer ${this.name} because it has not been built yet.`);if(null===this._refCount)throw Error(`Cannot dispose Layer ${this.name} because it has not been used yet.`);this.assertNotDisposed();let e=0;return 0==--this._refCount&&(e=this.disposeWeights()),{refCountAfterDispose:this._refCount,numDisposedVariables:e}}}class sy extends sx{constructor(e){if(super({dtype:e.dtype,name:null!=e.name?e.name:n1("input").toString()}),null==e.batchSize&&(e.batchSize=null),null==e.sparse&&(e.sparse=!1),this.trainable=!1,this.built=!0,this.sparse=e.sparse,null!=e.inputShape&&null!=e.batchInputShape)throw new nF("Only provide the inputShape OR batchInputShape argument to inputLayer, not both at the same time.");let t=e.batchInputShape;if(null==t)if(null==e.inputShape)throw new nF("An InputLayer should be passed either a `batchInputShape` or an `inputShape`.");else t=[e.batchSize].concat(e.inputShape);else if(null!=e.batchSize)throw new nF("Cannot specify batchSize if batchInputShape is specified when creating an InputLayer.");const r=e.dtype||"float32";this.batchInputShape=t,this.dtype=r,this.inputSpec=[{shape:t}];const n=new sc(this.dtype,this.batchInputShape,this,[],{},this.name);n.nodeIndex=0,n.tensorIndex=0,new sm({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:[n],outputTensors:[n],inputMasks:[null],outputMasks:[null],inputShapes:[t],outputShapes:[t]})}apply(e,t){throw new nF(`Cannot pass any input to an InputLayer's apply() method. InputLayer name: ${this.name}`)}dispose(){return{refCountAfterDispose:this._refCount,numDisposedVariables:0}}getConfig(){return{batchInputShape:this.batchInputShape,dtype:this.dtype,sparse:this.sparse,name:this.name}}}function sb(e){if(null==e.batchShape&&null==e.shape)throw Error("Please provide to Input either a `shape` or a `batchShape` argument. Note that `shape` does not include the batch dimension.");if(null!=e.batchShape&&null!=e.shape)throw new nF("Please provide either a `shape` or `batchShape` argument to Input, but not both.");let t=e.batchShape;null!=e.shape&&null==t&&(t=[null].concat(e.shape));let r=e.dtype;return null==r&&(r="float32"),new sy({batchInputShape:t,name:e.name,dtype:r,sparse:e.sparse}).inboundNodes[0].outputTensors[0]}sy.className="InputLayer",nJ.registerClass(sy);class sv{constructor(e){if(this.id2Value={},this.id2Mask={},this.name2Id={},e instanceof sv)for(const t in e.id2Value)this.id2Value[t]=e.id2Value[t],t in e.id2Mask&&(this.id2Mask[t]=e.id2Mask[t]);else{if(null==e)return;for(const t of e)this.add(t.key,t.value)}}add(e,t,r){if(null==this.id2Value[e.id])this.id2Value[e.id]=function(e,t){if(null==e.dtype||e.dtype===t.dtype)return t;try{return(0,y.cast)(t,e.dtype)}catch(r){throw new nF(`The dtype of the feed (${t.dtype}) can not be cast to the dtype of the key '${e.name}' (${e.dtype}).`)}}(e,t),this.name2Id[e.name]=e.id,null!=r&&(this.id2Mask[e.id]=r);else throw new nF(`Duplicate key: name=${e.name}, id=${e.id}`);return this}addFeed(e){this.add(e.key,e.value)}hasKey(e){return null!=this.id2Value[e.id]}names(){return Object.keys(this.name2Id)}getValue(e){if(e instanceof sc)if(null!=this.id2Value[e.id])return this.id2Value[e.id];else throw new nF(`Nonexistent key: ${e.name}`);{let t=this.name2Id[e];if(null==t)throw new nF(`Feed dict has no SymbolicTensor name: ${e}`);return this.id2Value[t]}}getMask(e){if(e instanceof sc)if(null!=this.id2Value[e.id])return this.id2Mask[e.id];else throw new nF(`Nonexistent key: ${e.name}`);{let t=this.name2Id[e];if(null==t)throw new nF(`Feed dict has no SymbolicTensor name: ${e}`);return this.id2Mask[t]}}disposeMasks(){null!=this.id2Mask&&(0,n$.dispose)(this.id2Mask)}}let sw=new nL,sC=new nL;function sI(e,t,r,n){let a,s=null!=r&&r.training,i=Array.isArray(e),o=i?e:[e],l=o.map(e=>e.name),u=[],h=t.names();for(let e of l)-1!==h.indexOf(e)?u.push(t.getValue(e)):u.push(null);null!=n&&(n.maxNumTensors=-1/0,n.minNumTensors=1/0);let d=l.join(",")+"|"+t.names().sort().join(","),p=sw.get(d);if(null==p){let e=function(e,t){nR.util.assert(null!=e&&e.length>0,()=>"Expected at least one fetch, got none");let r=[],n={};if(1===e.length){let a=sk(e[0],t);r=a.sorted,n=a.recipientMap}else{let a=new Set;for(let s of e){let{sorted:e,recipientMap:i}=sk(s,t);for(let t of e)a.has(t.name)||(r.push(t),a.add(t.name));for(let e in i)null==n[e]&&(n[e]=new Set),i[e].forEach(t=>n[e].add(t))}}return{sorted:r,recipientCounts:function(e){let t={};for(let r in e)t[r]=e[r].size;return t}(n)}}(o,t);p=e.sorted,a=e.recipientCounts,sw.put(d,p),sC.put(d,a)}a={},s||Object.assign(a,sC.get(d));let c=new sv(t);for(let e=0;e<p.length;++e){if(null!=n){let e=(0,n$.memory)().numTensors;e>n.maxNumTensors&&(n.maxNumTensors=e),e<n.minNumTensors&&(n.minNumTensors=e)}let i=p[e],o=i.sourceLayer;if(o instanceof sy)continue;let h=[],d=[],f=[],m=!1;for(let e of i.inputs){let r=c.getValue(e),n=c.getMask(e);h.push(r),d.push(n),null!=n&&(m=!0),!s&&(a[e.name]--,0!==a[e.name]||t.hasKey(e)||-1!==l.indexOf(e.name)||r.isDisposed||!0===e.sourceLayer.stateful||f.push(r))}m&&((r=r||{}).mask=d[0]);let g=nB(o.apply(h,r)),x=null;o.supportsMasking&&(x=o.computeMask(h,d));let y=function(e){let t;if(1===e.sourceLayer.inboundNodes.length)t=e.sourceLayer.output;else{let r=null;for(let t=0;t<e.sourceLayer.inboundNodes.length;++t)for(let n of e.sourceLayer.inboundNodes[t].outputTensors)if(n.id===e.id){r=t;break}t=e.sourceLayer.getOutputAt(r)}return t}(i),b=Array.isArray(y)?y:[y];for(let e=0;e<b.length;++e){c.hasKey(b[e])||c.add(b[e],g[e],Array.isArray(x)?x[0]:x);let t=l.indexOf(b[e].name);-1!==t&&(u[t]=g[e])}s||(0,n$.dispose)(f)}return c.disposeMasks(),i?u:u[0]}function sk(e,t){let r=new Set,n=[],a={};for(let e of t.names())r.add(e);let s=[],i=[];for(s.push(e);s.length>0;){let e=s[s.length-1];if(r.has(e.name)){s.pop();continue}let t=i[i.length-1]===s.length-1;if(0===e.inputs.length||t)s.pop(),n.push(e),r.add(e.name),t&&i.pop();else for(let t of(i.push(s.length-1),e.inputs))null==a[t.name]&&(a[t.name]=new Set),a[t.name].add(e.name),r.has(t.name)||s.push(t)}return{sorted:n,recipientMap:a}}(0,nN.env)().registerFlag("TOPOLOGICAL_SORT_CACHE_MAX_ENTRIES",()=>100,function(e){null!=sw&&sw.setMaxEntries(e),null!=sC&&sC.setMaxEntries(e)});var nJ=nJ;function sS(e,t){return(0,n$.tidy)(()=>S.sqrt(F.sum(b.mul(e,e),t,!0)))}class sT extends nJ.Serializable{getConfig(){return{}}}class sN extends sT{constructor(e){super(),this.defaultMaxValue=2,this.defaultAxis=0,this.maxValue=null!=e.maxValue?e.maxValue:this.defaultMaxValue,this.axis=null!=e.axis?e.axis:this.defaultAxis}apply(e){return(0,n$.tidy)(()=>{let t=sS(e,this.axis),r=rE.clipByValue(t,0,this.maxValue);return b.mul(e,C.div(r,P.add(aN(),t)))})}getConfig(){return{maxValue:this.maxValue,axis:this.axis}}}sN.className="MaxNorm",nJ.registerClass(sN);class s$ extends sT{constructor(e){super(),this.defaultAxis=0,this.axis=null!=e.axis?e.axis:this.defaultAxis}apply(e){return(0,n$.tidy)(()=>C.div(e,P.add(aN(),sS(e,this.axis))))}getConfig(){return{axis:this.axis}}}s$.className="UnitNorm",nJ.registerClass(s$);class sR extends sT{apply(e){return nu.relu(e)}}sR.className="NonNeg",nJ.registerClass(sR);class sA extends sT{constructor(e){super(),this.defaultMinValue=0,this.defaultMaxValue=1,this.defaultRate=1,this.defaultAxis=0,this.minValue=null!=e.minValue?e.minValue:this.defaultMinValue,this.maxValue=null!=e.maxValue?e.maxValue:this.defaultMaxValue,this.rate=null!=e.rate?e.rate:this.defaultRate,this.axis=null!=e.axis?e.axis:this.defaultAxis}apply(e){return(0,n$.tidy)(()=>{let t=sS(e,this.axis),r=P.add(b.mul(this.rate,rE.clipByValue(t,this.minValue,this.maxValue)),b.mul(1-this.rate,t));return b.mul(e,C.div(r,P.add(aN(),t)))})}getConfig(){return{minValue:this.minValue,maxValue:this.maxValue,rate:this.rate,axis:this.axis}}}sA.className="MinMaxNorm",nJ.registerClass(sA);let sE={maxNorm:"MaxNorm",minMaxNorm:"MinMaxNorm",nonNeg:"NonNeg",unitNorm:"UnitNorm"};function sF(e,t={}){return nH(e,nJ.SerializationMap.getMap().classNameMap,t,"constraint")}function sD(e){return null==e?null:"string"==typeof e?sF({className:e in sE?sE[e]:e,config:{}}):e instanceof sT?e:sF(e)}e.s(["maxNorm",0,function(e){return new sN(e)},"minMaxNorm",0,function(e){return new sA(e)},"nonNeg",0,function(){return new sR},"unitNorm",0,function(e){return new s$(e)}],8408);var sO=e.i(8408);e.s(["constant",0,function(e){return new aY(e)},"glorotNormal",0,function(e){return new a3(e)},"glorotUniform",0,function(e){return new a2(e)},"heNormal",0,function(e){return new a4(e)},"heUniform",0,function(e){return new a5(e)},"identity",0,function(e){return new a0(e)},"leCunNormal",0,function(e){return new a6(e)},"leCunUniform",0,function(e){return new a8(e)},"ones",0,function(){return new aK},"orthogonal",0,function(e){return new a7(e)},"randomNormal",0,function(e){return new aJ(e)},"randomUniform",0,function(e){return new aZ(e)},"truncatedNormal",0,function(e){return new aQ(e)},"varianceScaling",0,function(e){return new a1(e)},"zeros",0,function(){return new aX}],97656);var sL=e.i(97656),sz=e.i(12113);async function s_(e){if(null==e)return;let t=[],r=[],n=[];for(let a in e){let s=e[a];"number"!=typeof s&&(t.push(s.data()),r.push(a),n.push(s))}if(t.length>0){let a=await Promise.all(t);for(let t=0;t<a.length;++t)e[r[t]]=a[t][0];(0,n$.dispose)(n)}}function sM(e){if(null!=e)for(let t in e){let r=e[t];"number"!=typeof r&&r.dispose()}}(s=d||(d={}))[s.SILENT=0]="SILENT",s[s.VERBOSE=1]="VERBOSE";class sP{constructor(){this.validationData=null}setParams(e){this.params=e}async onEpochBegin(e,t){}async onEpochEnd(e,t){}async onBatchBegin(e,t){}async onBatchEnd(e,t){}async onTrainBegin(e){}async onTrainEnd(e){}setModel(e){}}class sB{constructor(e,t=10){null==e&&(e=[]),this.callbacks=e,this.queueLength=t}append(e){this.callbacks.push(e)}setParams(e){for(let t of this.callbacks)t.setParams(e)}setModel(e){for(let t of this.callbacks)t.setModel(e)}async onEpochBegin(e,t){for(let r of(null==t&&(t={}),this.callbacks))await r.onEpochBegin(e,t)}async onEpochEnd(e,t){for(let r of(null==t&&(t={}),this.callbacks))await r.onEpochEnd(e,t)}async onBatchBegin(e,t){for(let r of(null==t&&(t={}),this.callbacks))await r.onBatchBegin(e,t)}async onBatchEnd(e,t){for(let r of(null==t&&(t={}),this.callbacks))await r.onBatchEnd(e,t)}async onTrainBegin(e){for(let t of(null==e&&(e={}),this.callbacks))await t.onTrainBegin(e)}async onTrainEnd(e){for(let t of(null==e&&(e={}),this.callbacks))await t.onTrainEnd(e)}}class sW extends sP{constructor(){super()}async onEpochBegin(e){this.seen=0,this.totals={}}async onBatchEnd(e,t){null==t&&(t={});let r=null==t.size?0:t.size;for(let e in this.seen+=r,t){let n=t[e];if("number"==typeof n)this.totals.hasOwnProperty(e)||(this.totals[e]=0),this.totals[e]=this.totals[e]+n*r;else{let t;e in this.totals?t=this.totals[e]:this.totals[e]=0;let a=(0,n$.tidy)(()=>(0,P.add)(this.totals[e],(0,b.mul)(n,r)));this.totals[e]=a,null!=t&&t.dispose()}}}async onEpochEnd(e,t){if(null!=t)for(let e of this.params.metrics)null!=this.totals[e]&&("number"==typeof this.totals[e]?t[e]=this.totals[e]/this.seen:(0,n$.tidy)(()=>{let r=(0,b.mul)((0,C.div)(1,this.seen),this.totals[e]);t[e]=r,this.totals[e].dispose(),(0,n$.keep)(t[e])}))}}class sG extends sP{async onTrainBegin(e){this.epoch=[],this.history={}}async onEpochEnd(e,t){for(let r in null==t&&(t={}),this.epoch.push(e),t)null==this.history[r]&&(this.history[r]=[]),this.history[r].push(t[r])}async syncData(){let e=[],t=[],r=[];for(let n in this.history){let a=this.history[n];for(let s=0;s<a.length;++s)if("number"!=typeof a[s]){let i=a[s];e.push(i.data()),t.push(n),r.push(s)}}let n=await Promise.all(e);for(let e=0;e<n.length;++e)this.history[t[e]][r[e]].dispose(),this.history[t[e]][r[e]]=n[e][0]}}class sU extends sP{constructor(e,t){if(super(),this.currentEpoch=0,this.nowFunc=e.nowFunc,this.nextFrameFunc=e.nextFrameFunc||sz.nextFrame,this.yieldEvery=t||"auto","auto"===this.yieldEvery&&(this.yieldEvery=125),"never"===this.yieldEvery&&null!=e.onYield)throw Error("yieldEvery is `never` but you provided an `onYield` callback. Either change `yieldEvery` or remove the callback");nR.util.isNumber(this.yieldEvery)&&(this.maybeWait=function(e,t,r){let n,a=null!=r?r():nR.util.now();return(...s)=>{let i=null!=r?r():nR.util.now();return i-a<t?n:(a=i,n=e(...s))}}(this.maybeWait.bind(this),this.yieldEvery,this.nowFunc)),this.trainBegin=e.onTrainBegin,this.trainEnd=e.onTrainEnd,this.epochBegin=e.onEpochBegin,this.epochEnd=e.onEpochEnd,this.batchBegin=e.onBatchBegin,this.batchEnd=e.onBatchEnd,this.yield=e.onYield}async maybeWait(e,t,r){let n=[];null!=this.yield&&(await s_(r),n.push(this.yield(e,t,r))),n.push(this.nextFrameFunc()),await Promise.all(n)}async onEpochBegin(e,t){this.currentEpoch=e,null!=this.epochBegin&&(await s_(t),await this.epochBegin(e,t))}async onEpochEnd(e,t){let r=[];null!=this.epochEnd&&(await s_(t),r.push(this.epochEnd(e,t))),"epoch"===this.yieldEvery&&r.push(this.nextFrameFunc()),await Promise.all(r)}async onBatchBegin(e,t){null!=this.batchBegin&&(await s_(t),await this.batchBegin(e,t))}async onBatchEnd(e,t){let r=[];null!=this.batchEnd&&(await s_(t),r.push(this.batchEnd(e,t))),"batch"===this.yieldEvery?r.push(this.nextFrameFunc()):nR.util.isNumber(this.yieldEvery)&&r.push(this.maybeWait(this.currentEpoch,e,t)),await Promise.all(r)}async onTrainBegin(e){null!=this.trainBegin&&(await s_(e),await this.trainBegin(e))}async onTrainEnd(e){null!=this.trainEnd&&(await s_(e),await this.trainEnd(e))}}function sV(e,t){return(null==e&&(e={}),e instanceof sP)?[e]:Array.isArray(e)&&e[0]instanceof sP?e:nB(e).map(e=>new sU(e,t))}class sH{static registerCallbackConstructor(e,t){nR.util.assert(e>=0&&Number.isInteger(e),()=>`Verbosity level is expected to be an integer >= 0, but got ${e}`),sH.checkForDuplicate(t),null==sH.constructors[e]&&(sH.constructors[e]=[]),sH.constructors[e].push(t)}static checkForDuplicate(e){for(let t in sH.constructors)sH.constructors[+t].forEach(t=>{if(t===e)throw new nF("Duplicate callback constructor.")})}static clear(){sH.constructors={}}static createCallbacks(e){let t=[];for(let r in sH.constructors){let n=+r;e>=n&&t.push(...sH.constructors[n])}return t.map(e=>new e)}}function sq(e,t,r,n,a,s,i,o,l){let u=new sG,h=[new sW,...sH.createCallbacks(t)];null!=e&&h.push(...e),h.push(u);let d=new sB(h);return d.setParams({epochs:r,initialEpoch:n,samples:a,steps:s,batchSize:i,verbose:t,doValidation:o,metrics:l}),{callbackList:d,history:u}}sH.constructors={};var sj=e.i(74331),sX=e.i(33153),nJ=nJ,nJ=nJ;function sK(e,t={},r=!1){return nH(e,nJ.SerializationMap.getMap().classNameMap,t,"layer",r)}var sY=e.i(40722);function sZ(e,t){return(0,n$.tidy)(()=>{"float32"!==e.dtype&&(e=y.cast(e,"float32"));let r=F.sum(aB(e),t,!0),n=sY.fill(r.shape,aN()),a=S.sqrt(rh.maximum(r,n));return C.div(e,a)})}function sJ(e,t){return(0,n$.tidy)(()=>r6.mean(aB(N.sub(t,e)),-1))}function sQ(e,t){return(0,n$.tidy)(()=>r6.mean(rf.abs(N.sub(t,e)),-1))}function s0(e,t){return(0,n$.tidy)(()=>{let r=N.sub(e,t),n=rE.clipByValue(rf.abs(e),aN(),Number.MAX_VALUE),a=rf.abs(C.div(r,n));return b.mul(100,r6.mean(a,-1))})}function s1(e,t,r=!1){return(0,n$.tidy)(()=>{if(r)t=ny.softmax(t);else{let e=F.sum(t,t.shape.length-1,!0);t=C.div(t,e)}return t=rE.clipByValue(t,aN(),1-aN()),I.neg(F.sum(b.mul(y.cast(e,"float32"),tI.log(t)),t.shape.length-1))})}function s2(e,t,r=!1){return(0,n$.tidy)(()=>{let n,a=y.cast(tc.floor((n=[aI(e.shape)],E.reshape(e,n))),"int32"),s=(t=rE.clipByValue(t,aN(),1-aN())).shape;return s1(E.reshape(nn.oneHot(a,s[s.length-1]),s),t,r)})}function s3(e,t){return(0,n$.tidy)(()=>{let r;return r=rE.clipByValue(t,aN(),1-aN()),r=tI.log(C.div(r,N.sub(1,r))),r6.mean(function(e,t){if(!nR.util.arraysEqual(e.shape,t.shape))throw new nF(`logits and labels must have the same shape, but got shapes ${JSON.stringify(e.shape)} and ${JSON.stringify(t.shape)}`);return(0,n$.tidy)(()=>{let r=nu.relu(t),n=I.neg(rf.abs(t));return P.add(N.sub(r,b.mul(t,e)),r1.log1p(eL.exp(n)))})}(e,r),-1)})}function s4(e,t){return(0,n$.tidy)(()=>{let r=sZ(e,-1),n=sZ(t,-1),a=b.mul(r,n);return I.neg(F.sum(a,-1))})}let s5={meanSquaredError:sJ,meanAbsoluteError:sQ,meanAbsolutePercentageError:s0,meanSquaredLogarithmicError:function(e,t){return(0,n$.tidy)(()=>{let r=rE.clipByValue(t,aN(),Number.MAX_VALUE),n=tI.log(P.add(1,r)),a=rE.clipByValue(e,aN(),Number.MAX_VALUE),s=tI.log(P.add(1,a));return r6.mean(aB(N.sub(n,s)),-1)})},squaredHinge:function(e,t){return(0,n$.tidy)(()=>{let r=rh.maximum(0,N.sub(1,b.mul(e,t)));return r6.mean(aB(r),-1)})},hinge:function(e,t){return(0,n$.tidy)(()=>{let r=rh.maximum(0,N.sub(1,b.mul(e,t)));return r6.mean(r,-1)})},categoricalHinge:function(e,t){return(0,n$.tidy)(()=>{let r=F.sum(b.mul(e,t),-1),n=r5.max(b.mul(N.sub(1,e),t),-1);return rh.maximum(0,P.add(1,N.sub(n,r)))})},logcosh:function(e,t){return(0,n$.tidy)(()=>{let r=Math.log(2),n=N.sub(t,e),a=N.sub(P.add(n,nb.softplus(b.mul(-2,n))),r);return r6.mean(a,-1)})},categoricalCrossentropy:s1,sparseCategoricalCrossentropy:s2,binaryCrossentropy:s3,kullbackLeiblerDivergence:function(e,t){return(0,n$.tidy)(()=>{let r=rE.clipByValue(e,aN(),1),n=rE.clipByValue(t,aN(),1);return F.sum(b.mul(e,tI.log(C.div(r,n))),-1)})},poisson:function(e,t){return(0,n$.tidy)(()=>{let r=tI.log(P.add(aN(),t));return r6.mean(N.sub(t,b.mul(e,r)),-1)})},cosineProximity:s4};function s6(e){if("string"!=typeof e)return e;{if(e in s5)return s5[e];let t=`Unknown loss ${e}`;throw e.toLowerCase().includes("softmaxcrossentropy")&&(t=`Unknown loss ${e}. Use "categoricalCrossentropy" as the string name for tf.losses.softmaxCrossEntropy`),new nF(t)}}function s8(e,t){return(0,n$.tidy)(()=>{let r=b.mul(.5,na.onesLike(t)),n=aR(e1.greater(t,r),e.dtype);return r6.mean(e7.equal(e,n),-1)})}function s7(e,t){return(0,n$.tidy)(()=>aR(e7.equal(rv.argMax(e,-1),rv.argMax(t,-1)),"float32"))}function s9(e,t){return(0,n$.tidy)(()=>y.cast(F.sum(el.logicalAnd(e7.equal(e,1),e7.equal(t,1))),"float32"))}function ie(e,t){return(0,n$.tidy)(()=>{let r=s9(e,t),n=(0,n$.tidy)(()=>y.cast(F.sum(el.logicalAnd(e7.equal(e,0),e7.equal(t,1))),"float32")),a=P.add(r,n);return y.cast(eu.where(e1.greater(a,0),C.div(r,a),0),"float32")})}function it(e,t){return s3(e,t)}function ir(e,t){return e.rank===t.rank&&(e=nw.squeeze(e,[e.rank-1])),(t=rv.argMax(t,-1)).dtype!==e.dtype&&(t=y.cast(t,e.dtype)),y.cast(e7.equal(e,t),"float32")}let ia={binaryAccuracy:s8,categoricalAccuracy:s7,precision:ie,categoricalCrossentropy:s1,sparseCategoricalCrossentropy:s2,mse:sJ,MSE:sJ,mae:sQ,MAE:sQ,mape:s0,MAPE:s0,cosine:s4};function is(e){if(n_(null!==e,`Unknown LossOrMetricFn ${e}`),"string"==typeof e)return e;{let t;for(let r of Object.keys(s5))if(s5[r]===e){t=r;break}if(void 0!==t)return t;for(let r of Object.keys(ia))if(ia[r]===e){t=r;break}return void 0!==t?t:e.name}}var ii=e.i(70161);function io(e,t,r=!1){if(null==e||"object"!=typeof e||Object.getPrototypeOf(e)!==Object.prototype||!function e(t){if(null===t)return!0;if("object"==typeof t)if(Object.getPrototypeOf(t)===Object.prototype){for(let r of Object.keys(t))if("string"!=typeof r||!e(t[r]))return!1;return!0}else{if(!Array.isArray(t))return!1;for(let r of t)if(!e(r))return!1;return!0}{let e=typeof t;return"string"===e||"number"===e||"boolean"===e}}(e))throw Error("User-defined metadata is expected to be a JSON object, but is not.");if(r){let r=JSON.stringify(e);r.length>1048576&&console.warn(`User-defined metadata of model "${t}" is too large in size (length=${r.length} when serialized). It is not recommended to store such large objects in user-defined metadata. Please make sure its serialized length is <= 1048576.`)}}function il(e,t,r=console.log){let n="";for(let r=0;r<e.length;++r)r>0&&(n=n.slice(0,n.length-1)+" "),n+=e[r],n=n.slice(0,t[r]),n+=" ".repeat(t[r]-n.length);r(n)}function iu(e,t,r){return("inboundNodes"===e||"outputLayers"===e||"inputLayers"===e)&&0===t&&"string"==typeof r}function ih(e,t){if(null===e)return null;if("string"==typeof e)return nG(e);if("number"==typeof e||"boolean"==typeof e)return e;if(e instanceof Array){let r=[],n=e.length;for(let a=0;a<n;++a){let n=e[a];iu(t,a,n)?r.push(n):r.push(ih(n,t))}return r}{let t={};for(let r of Object.keys(e)){let n=e[r];if("name"===r&&"string"==typeof n)t[r]=n;else{let e=nG(r);t[e]=ih(n,e)}}return t}}let id="4.22.0";class ip extends sx{constructor(e){if(super({}),this.containerNodes=new Set,this.name=e.name,null==this.name){const e=this.getClassName().toLowerCase();this.name=n1(e)}if(this.supportsMasking=!1,this.trainable_=!0,Array.isArray(e.inputs)?this.inputs=e.inputs.slice():this.inputs=[e.inputs],Array.isArray(e.outputs)?this.outputs=e.outputs.slice():this.outputs=[e.outputs],nj(this.inputs).length!==this.inputs.length)throw new nF(`The list of inputs passed to the model is redundant. All inputs should only appear once. Found: ${this.inputs.map(e=>e.name)}`);for(const e of(nj(this.outputs).length!==this.outputs.length&&console.warn(`The list of outputs passed to the model is redundant. All outputs should only appear once. Found: ${this.outputs.map(e=>e.name)}`),this.inputLayers=[],this.inputLayersNodeIndices=[],this.inputLayersTensorIndices=[],this.outputLayers=[],this.outputLayersNodeIndices=[],this.outputLayersTensorIndices=[],this.layers=[],this.internalContainerRefs=[],this.outputs)){const t=e.sourceLayer,r=e.nodeIndex,n=e.tensorIndex;this.outputLayers.push(t),this.outputLayersNodeIndices.push(r),this.outputLayersTensorIndices.push(n)}for(const e of this.inputs){const t=e.sourceLayer,r=e.nodeIndex,n=e.tensorIndex;n_(0===r,"input layer has >1 nodes"),n_(0===n,"input layer has >1 tensors"),this.inputLayers.push(t),this.inputLayersNodeIndices.push(r),this.inputLayersTensorIndices.push(n)}this.inputNames=[],this.outputNames=[],this.feedInputShapes=[],this.feedInputNames=[],this.feedOutputNames=[];for(let t=0;t<this.inputLayers.length;t++){const r=this.inputLayers[t];if(!(r instanceof sy))throw TypeError(`Input layers to a LayersModel must be InputLayer objects. Received inputs: ${e.inputs}. Input ${t} (0-based) originates from layer type ${r.getClassName()}.`);this.inputNames.push(r.name),this.feedInputShapes.push(r.batchInputShape),this.feedInputNames.push(r.name)}for(const e of this.outputLayers)this.outputNames.push(e.name);this.internalInputShapes=this.inputs.map(e=>e.shape),this.internalOutputShapes=this.outputs.map(e=>e.shape);const t={},r={},n={},a={},s={},i=[],o=(e,t,r,n,a,l)=>{(null==n||null==a||null==l)&&(n=e.sourceLayer,a=e.nodeIndex,l=e.tensorIndex);let u=n.inboundNodes[a];if(-1!==r.indexOf(u))throw new nE(`The tensor ${e.name} at layer "${n.name}" is part of a cycle.`);if(-1!==t.indexOf(u))return;this.containerNodes.add(ip.nodeKey(n,a)),n.id in s||(s[n.id]=Object.keys(s).length),-1===r.indexOf(u)&&r.push(u);let h=u.inboundLayers.length;for(let e=0;e<h;e++)o(u.inputTensors[e],t,r,u.inboundLayers[e],u.nodeIndices[e],u.tensorIndices[e]);for(t.push(u);r.indexOf(u)>=0;)r.splice(r.indexOf(u),1);i.push(u)},l=[],u=[];for(const e of this.outputs)o(e,l,u);for(const e of i.slice().reverse()){r[e.id]=e,e.id in t||(t[e.id]=0);let s=t[e.id];s=Math.max(s,null==n[e.outboundLayer.id]?0:n[e.outboundLayer.id]),n[e.outboundLayer.id]=s,a[e.outboundLayer.id]=e.outboundLayer,t[e.id]=s;for(let n=0;n<e.inboundLayers.length;n++){const a=e.inboundLayers[n],i=e.nodeIndices[n],o=a.inboundNodes[i],l=null==t[o.id]?0:t[o.id];t[o.id]=Math.max(s+1,l),r[o.id]=o}}const h={};for(const e in t){const n=t[e];n in h||(h[n]=[]),h[n].push(r[e])}const d={};for(const e in n){const t=n[e];t in d||(d[t]=[]),d[t].push(a[e])}let p=Object.keys(d).map(e=>parseInt(e,10)).sort(nq);for(const e of(this.layers=[],p)){const t=d[e];for(const e of(t.sort((e,t)=>{let r=s[e.id],n=s[t.id];return r<n?-1:+(r>n)}),t))e instanceof ip&&this.internalContainerRefs.push(e),this.layers.push(e)}this.layersByDepth=d,p=Object.keys(h).map(e=>parseInt(e,10)).sort(nq);const c=this.inputs.slice(),f=[];for(const e of p)for(const t of h[e]){const e=t.outboundLayer;if(null!=e){for(const r of t.inputTensors)if(-1===c.indexOf(r))throw new nE(`Graph disconnected: cannot obtain value for tensor ${r} at layer "${e.name}". The following previous layers were accessed without issue: ${f}`);for(const e of t.outputTensors)c.push(e);f.push(e.name)}}this.nodesByDepth=h;const m=this.layers.map(e=>e.name);for(const e of m){const t=m.filter(t=>t===e).length;if(1!==t)throw new nE(`The name "${e}" is used ${t} times in the model. All layer names should be unique. Layer names: `+JSON.stringify(m))}this.outboundNodes=[],this.inboundNodes=[],new sm({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:this.inputs.map(e=>null),outputMasks:this.outputs.map(e=>null),inputShapes:this.inputs.map(e=>e.shape),outputShapes:this.outputs.map(e=>e.shape)}),this.built=!0,this._refCount=1}assertNotDisposed(){if(0===this._refCount)throw Error(`Container '${this.name}' is already disposed.`)}dispose(){this.assertNotDisposed();let e={refCountAfterDispose:null,numDisposedVariables:0};if(0==--this._refCount){for(let t of this.layers)e.numDisposedVariables+=t.dispose().numDisposedVariables;for(let t of this.internalContainerRefs)e.numDisposedVariables+=t.dispose().numDisposedVariables}return e.refCountAfterDispose=this._refCount,e}get trainable(){return this.trainable_}set trainable(e){this.layers.forEach(t=>{t._trainableWeights.forEach(t=>t.trainable=e)}),this.trainable_=e}get trainableWeights(){if(this._trainableWeights.length>0)throw new nF("Container instance unexpectedly contains _trainableWeights.The trainable weights of a Container are a union of the trainable weights of its consituent Layers. Its own _trainableWeights must remain an empty Array.");if(!this.trainable)return[];let e=[];for(let t of this.layers)e=e.concat(t.trainableWeights);return e}get nonTrainableWeights(){let e=[];for(let t of this.layers)e.push(...t.nonTrainableWeights);if(!this.trainable){let t=[];for(let e of this.layers)t.push(...e.trainableWeights);return t.concat(e)}return e}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}loadWeights(e,t=!0){let r={},n=0,a=(e=>{let t=Object.keys(e);if(0===t.length)return!1;let r=t[0].split("/");return!isNaN(parseInt(r[r.length-1],10))})(e);for(let t of(a&&this.parseWeights(e),this.layers))for(let[e,s]of t.weights.entries()){let t=a?`${s.name.split("/").slice(0,-1).join("/")+"/"}${e}`:s.originalName;if(null!=r[t])throw new nF(`Duplicate weight name: ${t}`);r[t]=s,n++}let s=[];for(let n in e){let a=n;if(null==r[n]){let e=n.split("/");a=e.slice(0,-2).concat([e[e.length-1]]).join("/")}if(null!=r[a])s.push([r[a],e[n]]);else if(t)throw new nF(`Provided weight data has no target variable: ${n}`);delete r[a]}if(t){let e=[];for(let t in r)e.push(t);if(e.length>0)throw new nF(`${e.length} of ${n} weights are not set: ${e}`)}sd(s)}parseWeights(e){for(let t in Object.keys(e)){let r=t.split("/"),n=["vars","layer_checkpoint_dependencies"],a=r.map(e=>e.startsWith("_")?e.slice(1):e).filter(e=>!n.includes(e)).join("/");a!==t&&(e[a]=e[t],delete e[t])}}updatedConfig(){let e=this.getConfig(),t={};return t.className=this.getClassName(),t.config=e,t.kerasVersion=`tfjs-layers ${id}`,t.backend="TensorFlow.js",t}toJSON(e,t=!0){let r=function e(t,r){if(null==t)return null;if("string"==typeof t)return nW(t);if("number"==typeof t||"boolean"==typeof t)return t;if(t instanceof Array){let n=[],a=t.length;for(let s=0;s<a;++s){let a=t[s];iu(r,s,a)?n.push(a):n.push(e(a,r))}return n}{let r={};for(let n of Object.keys(t)){let a=t[n],s=nW(n);("name"===n||"className"===n)&&"string"==typeof a?r[s]=a:r[s]=e(a,n)}return r}}(this.updatedConfig());return t?JSON.stringify(r):r}call(e,t){return(0,n$.tidy)(()=>{e=nB(e);let r=new sv;for(let t=0;t<this.inputs.length;++t)r.add(this.inputs[t],e[t]);return sI(this.outputs,r,t)})}computeMask(e,t){return(0,n$.tidy)(()=>{let r;return e=nB(e),r=null==t?nz(null,e.length):nB(t),this.runInternalGraph(e,r)[1]})}computeOutputShape(e){let t=sn(e);if(t.length!==this.inputLayers.length)throw new nF(`Invalid inputShape argument ${e}: model has ${this.inputLayers.length} tensor inputs.`);let r={};for(let e=0;e<t.length;e++){let n=this.inputLayers[e],a=t[e];r[n.name+"_0_0"]=a}let n=Object.keys(this.nodesByDepth).map(e=>parseInt(e,10)).sort(nq);if(n.length>1)for(let e of n)for(let t of this.nodesByDepth[e]){let e=t.outboundLayer;if(-1!==this.inputLayers.map(e=>e.id).indexOf(e.id))continue;let n=[];for(let e=0;e<t.inboundLayers.length;e++){let a=t.inboundLayers[e],s=t.nodeIndices[e],i=t.tensorIndices[e],o=r[`${a.name}_${s}_${i}`];n.push(o)}let a=sn(e.computeOutputShape(nP(n))),s=e.inboundNodes.indexOf(t);for(let t=0;t<a.length;t++)r[`${e.name}_${s}_${t}`]=a[t]}let a=[],s=[];for(let e=0;e<this.outputLayers.length;e++){let t=this.outputLayers[e],r=this.outputLayersNodeIndices[e],n=this.outputLayersTensorIndices[e],a=`${t.name}_${r}_${n}`;s.push(a)}for(let e=0;e<s.length;e++){let t=s[e];n_(t in r),a.push(r[t])}return nP(a)}runInternalGraph(e,t){null==t&&(t=nz(null,e.length));let r={};for(let n=0;n<this.inputs.length;++n){let a=this.inputs[n],s=e[n],i=t[n];r[a.id]=[s,i]}for(let e of Object.keys(this.nodesByDepth).map(e=>parseInt(e,10)).sort(nq))for(let t of this.nodesByDepth[e]){let e=t.outboundLayer,n=t.inputTensors,a=t.outputTensors,s=[];for(let e of n)e.id in r&&s.push(r[e.id]);if(s.length===n.length){let n,i,o,l,u={};if(null!=t.callArgs&&(u=t.callArgs),1===s.length){let[t,r]=s[0];null==u.mask&&(u.mask=r),o=nB(e.call(t,u)),l=nB(e.computeMask(t,r)),n=[t],i=[r]}else n=s.map(e=>e[0]),i=s.map(e=>e[1]),null==u.mask&&(u.mask=i),o=nB(e.call(n,u)),l=nB(e.computeMask(n,i));if(e.activityRegularizer)throw new nD("LayersModel invocation with concrete Tensor value(s) in the presence of activity regularizer(s) is not supported yet.");for(let e=0;e<a.length;++e){let t=a[e],n=o[e],s=l[e];r[t.id]=[n,s]}}}let n=[],a=[],s=[];for(let e of this.outputs){n_(e.id in r,`Could not compute output ${e.name} : ${e.id}`);let[t,i]=r[e.id];s.push(t.shape),n.push(t),a.push(i)}return[n,a,s]}buildNodeConversionMap(e){let t,r={};for(let e of this.layers){t=+(e instanceof ip);for(let n=0;n<e.inboundNodes.length;n++){let a=ip.nodeKey(e,n);this.containerNodes.has(a)&&(r[a]=t,t+=1)}}return r}getLayer(e,t){if(null!=t)return this.findLayer(t);if(null==e)throw new nF("Provide either a layer name or layer index");if("number"==typeof e)return this.findLayer(e);for(let t of this.layers)if(t.name===e)return t;throw new nF(`No such layer: ${e}`)}findLayer(e){if(!(this.layers.length<=e))return this.layers[e];throw new nF(`Was asked to retrieve layer at index ${e}, but model only has ${this.layers.length} layer(s).`)}calculateLosses(){return(0,n$.tidy)(()=>{let e=[];for(let t of this.layers)for(let r=0;r<t.inboundNodes.length;++r){let n=ip.nodeKey(t,r);this.containerNodes.has(n)&&e.push(...t.calculateLosses())}return e})}getConfig(){let e={name:this.name},t=this.buildNodeConversionMap(this.layers),r=[];for(let e of this.layers){let n=e.getClassName(),a=e.getConfig(),s=[];for(let r=0;r<e.inboundNodes.length;r++){let n=e.inboundNodes[r],a=ip.nodeKey(e,r),i={};if(this.containerNodes.has(a)){if(n.callArgs)try{JSON.stringify(n.callArgs),i=n.callArgs}catch(t){console.warn(`Layer ${e.name} was passed non-serializable keyword arguments: ${n.callArgs}. They will not be included in the serialized model (and thus will be missing at deserialization time).`),i={}}if(n.inboundLayers.length>0){let e=[];for(let r=0;r<n.inboundLayers.length;r++){let a=n.inboundLayers[r],s=n.nodeIndices[r],o=n.tensorIndices[r],l=t[ip.nodeKey(a,s)];null==l&&(l=0),e.push([a.name,l,o,i])}s.push(e)}}}let i={};i.name=e.name,i.className=n,i.config=a,i.inboundNodes=s,r.push(i)}e.layers=r;let n=[];for(let e=0;e<this.inputLayers.length;e++){let r=this.inputLayers[e],a=this.inputLayersNodeIndices[e],s=ip.nodeKey(r,a);if(!this.containerNodes.has(s))continue;let i=t[s];null==i&&(i=0);let o=this.inputLayersTensorIndices[e];n.push([r.name,i,o])}e.inputLayers=n;let a=[];for(let e=0;e<this.outputLayers.length;e++){let r=this.outputLayers[e],n=this.outputLayersNodeIndices[e],s=ip.nodeKey(r,n);if(!this.containerNodes.has(s))continue;let i=t[s];null==i&&(i=0);let o=this.outputLayersTensorIndices[e];a.push([r.name,i,o])}return e.outputLayers=a,e}static fromConfig(e,t,r={},n=!1){let a={},s={};function i(e,t){e.name in s?s[e.name].push(t):s[e.name]=[t]}let o=t.name,l=t.layers;for(let e of l)!function(e){let r=e.name,s=sK(e,null!=t.customObjects?t.customObjects:{});s.setFastWeightInitDuringBuild(n),a[r]=s,e.inboundNodes.forEach(e=>{if(!(e instanceof Array))throw new nF(`Corrupted configuration, expected array for nodeData: ${e}`);i(s,e)})}(e);for(;!function(e){if(null==e)throw new nF(`Invalid value in obj: ${JSON.stringify(e)}`);for(let t in e)if(e.hasOwnProperty(t))return!1;return!0}(s);)for(let e of l){let t=a[e.name];if(t.name in s){let e=s[t.name];for(let r of(delete s[t.name],e))!function(e,t){let r,n=[];for(let s of t){let o=s[0],l=s[1],u=s[2];if(r=null==s[3]?{}:s[3],!(o in a))return void i(e,t);let h=a[o];if(h.inboundNodes.length<=l)return void i(e,t);let d=h.inboundNodes[l];n.push(d.outputTensors[u])}n.length>0&&e.apply(nP(n),r)}(t,r)}}let u=[],h=[];for(let e of t.inputLayers){let t=e[0],r=e[1],n=e[2];n_(t in a);let s=a[t].inboundNodes[r].outputTensors;u.push(s[n])}for(let e of t.outputLayers){let t=e[0],r=e[1],n=e[2];n_(t in a);let s=a[t].inboundNodes[r].outputTensors;h.push(s[n])}return new e({inputs:u,outputs:h,name:o})}get stateful(){if(this._stateful)throw new nF("Container instance unexpectedly has _stateful = true. The statefulness of a Container is determined by the Layers it contains. Its _stateful property must remain the default false.");for(let e of this.layers)if(e.stateful)return!0;return!1}resetStates(){(0,n$.tidy)(()=>{this.layers.forEach(e=>{e.stateful&&e.resetStates()})})}}var ic=e.i(67207);function im(e,t){var r="classWeight";let n=t.length;if(null==e||Array.isArray(e)&&0===e.length)return t.map(e=>null);if(1===n)if(Array.isArray(e)&&1===e.length)return e;else if("object"==typeof e&&t[0]in e)return[e[t[0]]];else return[e];if(Array.isArray(e)){if(e.length!==n)throw Error(`Provided ${r} is an array of ${e.length} element(s), but the model has ${n} outputs. Make sure a set of weights is provided for each model output.`);return e}if("object"==typeof e&&Object.keys(e).length>0&&"object"==typeof e[Object.keys(e)[0]]){let r=[];return t.forEach(t=>{t in e?r.push(e[t]):r.push(null)}),r}throw Error(`The model has multiple (${n}) outputs, so ${r} must be either an array with ${n} elements or an object with ${t} keys. Provided ${r} not understood: ${JSON.stringify(e)}`)}async function ig(e,t,r,n){if(null!=t||null!=n)throw Error("Support sampleWeight is not implemented yet");if(null==r)return null;{let t=(0,n$.tidy)(()=>{if(1===e.shape.length)return(0,ic.clone)(e);if(2===e.shape.length)if(e.shape[1]>1)return(0,rv.argMax)(e,1);else if(1===e.shape[1])return(0,E.reshape)(e,[e.shape[0]]);else throw Error(`Encountered unexpected last-dimension size (${e.shape[1]}) during handling of class weights. The size is expected to be >= 1.`);throw Error(`Unexpected rank of target (y) tensor (${e.rank}) during handling of class weights. The rank is expected to be 1 or 2.`)}),n=Array.from(await t.data());(0,n$.dispose)(t);let a=[];return n.forEach(e=>{if(null==r[e])throw Error(`classWeight must contain all classes in the training data. The class ${e} exists in the data but not in classWeight`);a.push(r[e])}),(0,aC.tensor1d)(a,"float32")}}function ix(e,t){let r,n;r=t.xs,n=t.ys,nR.util.assert(null!=r&&null!=n,()=>`A Dataset iterator for fitDataset() is expected to generate objects of the form \`{xs: xVal, ys: yVal}\`, where the two values may be \`tf.Tensor\`, an array of Tensors, or a map of string to Tensor.  The provided Dataset instead generates ${t}`);let a=iy("input",e.inputNames,r),s=iy("output",e.outputNames,n),i=a[0].shape[0];nR.util.assert(a.length===e.inputs.length,()=>`LayersModel has ${e.inputs.length} inputs, but the dataset provides ${a.length} inputs.  (Expected input keys: ${JSON.stringify(e.inputNames)})`),nR.util.assert(s.length===e.outputs.length,()=>`LayersModel has ${e.outputs.length} outputs, but the dataset provides ${s.length} outputs.  (Expected output keys: ${JSON.stringify(e.outputNames)})`);for(let t=0;t<a.length;t++)nR.util.assert(a[t].shape[0]===i,()=>`Batch size mismatch: input ${e.inputNames[t]} has ${a[t].shape[0]}; expected  ${i} based on input ${e.inputNames[0]}.`);for(let t=0;t<s.length;t++)nR.util.assert(s[t].shape[0]===i,()=>`Batch size mismatch: output ${e.outputNames[t]} has ${s[t].shape[0]}; expected  ${i} based on input ${e.inputNames[0]}.`);return{xs:a,ys:s}}function iy(e,t,r){if(r instanceof rm.Tensor)return[r];if(Array.isArray(r))return nR.util.assert(r.length===t.length,()=>`Received an array of ${r.length} Tensors, but expected ${t.length} to match the ${e} keys ${t}.`),r;{let n=[];for(let a of t){if(null==r[a])throw new nF(`The feature data generated by the dataset lacks the required ${e} key '${a}'.`);n.push(r[a])}return n}}async function ib(e,t,r){let n=null!=r.batchesPerEpoch;if(nR.util.assert(null!=e.optimizer,()=>"You must compile a model before training/testing. Use LayersModel.compile(modelCompileConfig)."),nR.util.assert(null!=r,()=>"For fitDataset(), the 2nd argument (config) is required, but it is not provided in this call."),nR.util.assert(null!=r.epochs&&r.epochs>0&&Number.isInteger(r.epochs),()=>`For fitDataset(), config.epochs is expected to be a positive integer, but got ${r.epochs}`),nR.util.assert(!n||r.batchesPerEpoch>0&&Number.isInteger(r.batchesPerEpoch),()=>`For fitDataset(), config.batchesPerEpoch is expected to be a positive integer if specified, but got ${r.batchesPerEpoch}`),nR.util.assert(null==r.validationSplit,()=>"`validationSplit` is not supported by `fitDataset()`. Use validationData instead."),e.isTraining)throw Error("Cannot start training because another fit() call is ongoing.");e.isTraining=!0;try{var a,s;let i,o,l,u,h=null!=r.validationData;if(h)if(iv(r.validationData))nR.util.assert(null==r.validationBatches||r.validationBatches>0&&Number.isInteger(r.validationBatches),()=>`For fitDataset() with dataset-based validation, config.validationBatches is expected not to be provided, or to be a positive integer, but got ${r.validationBatches}`);else{let e=function(e){if(3===e.length)throw new nD("Validation with sample weights is not implemented yet.");return{xs:e[0],ys:e[1]}}(r.validationData);o=e.xs,l=e.ys}let d=e.makeTrainFunction(),p=e.getDedupedMetricsNames();u=h?p.slice().concat(p.map(e=>"val_"+e)):p.slice();let c=sV(r.callbacks,r.yieldEvery),f=null==r.verbose?1:r.verbose,{callbackList:m,history:g}=sq(c,f,r.epochs,null,null,(a=t,s=r,i=null,null!=s.batchesPerEpoch?i=s.batchesPerEpoch:Number.isFinite(a.size)&&(i=a.size),i),null,h,u);m.setModel(e),e.history=g,await m.onTrainBegin(),e.stopTraining_=!1;let x=null==r.initialEpoch?0:r.initialEpoch,y=await t.iterator();for(;x<r.epochs;){let a={};await m.onEpochBegin(x);let s=0,i=0;for(n||(y=await t.iterator());!n||s<r.batchesPerEpoch;){let t=await y.next();if(n&&t.done){console.warn(`You provided \`batchesPerEpoch\` as ${r.batchesPerEpoch}, but your dataset iterator ran out of data after ${s} batches; interrupting training. Make sure that your dataset can generate at least \`batchesPerEpoch * epochs\` batches (in this case, ${r.batchesPerEpoch*r.epochs} batches). You may need to use the repeat() function when building your dataset.`);break}if(null!=t.value){let{xs:n,ys:a}=ix(e,t.value),o={};o.batch=i,o.size=n[0].shape[0],await m.onBatchBegin(i,o);let l=[];if(null!=r.classWeight){let t=im(r.classWeight,e.outputNames);for(let e=0;e<t.length;++e)l.push(await ig(a[e],null,t[e]))}let u=n.concat(a).concat(l),h=d(u);n$.dispose(u);for(let e=0;e<p.length;++e){let t=p[e],r=h[e];o[t]=r,n$.keep(r)}await m.onBatchEnd(i,o),sM(o),i++,s++}if(n?s>=r.batchesPerEpoch:t.done){if(h){let t;t=iv(r.validationData)?nB(await e.evaluateDataset(r.validationData,{batches:r.validationBatches})):nB(e.evaluate(o,l,{batchSize:null==r.validationBatchSize?32:r.validationBatchSize,verbose:0}));for(let r=0;r<e.metricsNames.length;++r)a[`val_${e.metricsNames[r]}`]=t[r]}break}if(e.stopTraining_)break}if(await m.onEpochEnd(x,a),x++,e.stopTraining_)break}return await m.onTrainEnd(),await e.history.syncData(),e.history}finally{e.isTraining=!1}}function iv(e){return"function"==typeof e.iterator}async function iw(e,t,r){let n=null!=(r=r||{}).batches,a=e.testFunction,s=[];if(r.verbose>0)throw new nD("Verbose mode is not implemented yet.");nR.util.assert(!n||r.batches>0&&Number.isInteger(r.batches),()=>`Test loop expects \`batches\` to be a positive integer, but received ${JSON.stringify(r.batches)}`);let i="function"==typeof t.next?t:await t.iterator(),o=0,l=0;for(;!n||l<r.batches;){let t=await i.next();if(s=n$.tidy(()=>{if(t.value){let{xs:r,ys:n}=ix(e,t.value),i=r.concat(n),u=n$.tidy(()=>a(i));if(n$.dispose(i),0===l)for(let e=0;e<u.length;++e)s.push((0,k.scalar)(0));let h=i[0].shape[0];for(let e=0;e<u.length;++e){let t=u[e],r=s[e];s[e]=n$.tidy(()=>P.add(s[e],b.mul(h,t))),l>0&&n$.dispose(r)}n$.dispose(u),o+=h,++l}return s}),t.done){n&&console.warn(`Your dataset iterator ran out of data during evaluateDataset(). Interrupting evalution. Make sure that your dataset can generate at least \`batches\` batches (in this case, ${r.batches} batches). You may need to use the repeat() function when building your dataset.`);break}}for(let e=0;e<s.length;++e){let t=s[e];s[e]=C.div(s[e],o),n$.dispose(t)}return nP(s)}function iC(e){nR.util.assert(e>0&&Number.isInteger(e),()=>`batchSize is required to be a positive integer, but got ${e}`)}function iI(e,t,r){return null==e?[null]:Array.isArray(e)?e.map(e=>aE(e,t,r-t)):aE(e,t,r-t)}function ik(e,t){return n$.tidy(()=>null==e?null:Array.isArray(e)?e.map(e=>ik(e,t)):aP(e,"int32"===t.dtype?t:y.cast(t,"int32")))}function iS(e,t){let r=[],n=0,a=null;for(;n<e;)(a=n+t)>=e&&(a=e),r.push([n,a]),n=a;return r}function iT(e){let t=[];e instanceof rm.Tensor&&(e=[e]);for(let r=0;r<e.length;++r){let n=e[r];if(1===n.rank)t.push(aA(n,1));else if(0===n.rank)throw Error("Expected tensor to be at least 1D, but received a 0D tensor (scalar).");else t.push(n)}return t}function iN(e,t){if(null==e)return;let r=[];if(t instanceof rm.Tensor)r.push(t.id);else if(Array.isArray(t))t.forEach(e=>r.push(e.id));else if(null!=t)for(let e in t){let n=t[e];r.push(n.id)}let n=[];if(e instanceof rm.Tensor)-1===r.indexOf(e.id)&&n.push(e);else if(Array.isArray(e))e.forEach(e=>{-1===r.indexOf(e.id)&&n.push(e)});else if(null!=e)for(let t in e){let a=e[t];-1===r.indexOf(a.id)&&n.push(a)}n.forEach(e=>{e.isDisposed||e.dispose()})}function i$(e){return Array.isArray(e)}function iR(e){return!(e instanceof rm.Tensor)&&!i$(e)}function iA(e,t,r,n=!0,a=""){let s;if(null==t||0===t.length){if(null!=e){let t=!1;if(i$(e)&&e.length>0)t=!0;else if(iR(e)){for(let r in e)if(e.hasOwnProperty(r)){t=!0;break}}else t=!0;if(t)throw new nF(`Error when checking model ${a} expected no data, but got ${e}`)}return[]}if(null==e)return t.map(e=>null);if(iR(e))for(let r of(s=[],t)){if(null==e[r])throw new nF(`No data provided for "${r}". Need data for each key in: ${t}`);s.push(e[r])}else if(i$(e)){if(e.length!==t.length)throw new nF(`Error when checking model ${a}: the Array of Tensors that you are passing to your model is not the size the model expected. Expected to see ${t.length} Tensor(s), but instead got the following list of Tensor(s): ${e}`);s=e}else{if(t.length>1)throw new nF(`The model ${a} expects ${t.length} Tensor(s), but only received one Tensor. Found: Tensor with shape ${e.shape}`);s=[e]}if(s=iT(s),null!=r)for(let e=0;e<t.length;++e){if(null==r[e])continue;let i=s[e];if(i.shape.length!==r[e].length)throw new nF(`Error when checking ${a}: expected ${t[e]} to have ${r[e].length} dimension(s). but got array with shape ${i.shape}`);for(let t=0;t<r[e].length;++t){if(0===t&&!n)continue;let s=i.shape[t],o=r[e][t];if(null!=o&&o>=0&&s!==o)throw new nF(`${a} expected a batch of elements where each example has shape [${r[e].slice(1,r[e].length)}] (i.e.,tensor shape [*,${r[e].slice(1,r[e].length)}]) but the ${a} received an input with ${i.shape[0]} examples, each with shape [${i.shape.slice(1,i.shape.length)}] (tensor shape [${i.shape}])`)}}return s}function iE(e,t,r,n=!0,a=""){let s;if(Array.isArray(e)){if(e.length!==t.length)throw new nF(`Error when checking model ${a}: the Array of Tensors that you are passing to your model is not the size the the model expected. Expected to see ${t.length} Tensor(s), but instead got ${e.length} Tensors(s).`);s=e}else{if(t.length>1)throw new nF(`The model expects ${t.length} ${a} Tensors, but only received one Tensor. Found: array with shape ${JSON.stringify(e.shape)}.`);s=[e]}if(null!=r)for(let e=0;e<t.length;++e){if(null==r[e])continue;let i=s[e];if(i.shape.length!==r[e].length)throw new nF(`Error when checking ${a}: expected ${t[e]} to have ${r[e].length} dimension(s), but got array with shape ${JSON.stringify(i.shape)}`);for(let s=0;s<r[e].length;++s){if(0===s&&!n)continue;let o=i.shape[s],l=r[e][s];if(null!=l&&l!==o)throw new nF(`Error when checking ${a}: expected ${t[e]} to have shape ${JSON.stringify(r[e])} but got array with shape ${JSON.stringify(i.shape)}.`)}}}class iF extends ip{constructor(e){super(e),this.isTraining=!1}summary(e,t,r=console.log){if(!this.built)throw new nF("This model has never been called, thus its weights have not been created yet. So no summary can be displayed. Build the model first (e.g., by calling it on some test data).");!function(e,t,r,n=console.log){var a;let s,i=function(e){let t=!0,r=[],n=[];for(let t in e.nodesByDepth)r.push(e.nodesByDepth[t]);for(let e of r){if(e.length>1||1===e.length&&e[0].inboundLayers.length>1){t=!1;break}n.push(...e)}if(t)for(let r of e.layers){let e=!1;for(let a of r.inboundNodes)if(-1!==n.indexOf(a))if(e){t=!1;break}else e=!0;if(!t)break}return t}(e),o=["Layer (type)","Input Shape","Output shape","Param #"];if(i?(t=t||90,r=r||[.32,.61,.89,1]):(t=t||115,r=r||[.24,.48,.7,.8,1]),r[r.length-1]<=1&&(r=r.map(e=>Math.floor(t*e))),!i)for(let t in o.push("Receives inputs"),s=[],e.nodesByDepth)s.push(...e.nodesByDepth[t]);n("_".repeat(t)),il(o,r,n),n("=".repeat(t));let l=e.layers;for(let e=0;e<l.length;++e)i?function(e,t,r){let n,a;try{a=e.inboundNodes.map(e=>JSON.stringify(e.inputShapes)).join(",")}catch(e){a="multiple"}try{n=JSON.stringify(e.outputShape)}catch(e){n="multiple"}let s=e.name,i=e.getClassName();il([`${s} (${i})`,a,n,e.countParams().toString()],t,r)}(l[e],r,n):function(e,t,r,n){let a,s;try{s=e.inboundNodes.map(e=>JSON.stringify(e.inputShapes)).join(",")}catch(e){s="multiple"}try{a=JSON.stringify(e.outputShape)}catch(e){a="multiple"}let i=[];for(let t of e.inboundNodes)if(null==r||!(r.length>0)||-1!==r.indexOf(t))for(let e=0;e<t.inboundLayers.length;++e){let r=t.inboundLayers[e].name,n=t.nodeIndices[e],a=t.tensorIndices[e];i.push(`${r}[${n}][${a}]`)}let o=e.name,l=e.getClassName(),u=0===i.length?"":i[0];il([`${o} (${l})`,s,a,e.countParams().toString(),u],t,n);for(let e=1;e<i.length;++e)il(["","","","",i[e]],t,n)}(l[e],r,s,n),n((e===l.length-1?"=":"_").repeat(t));e.checkTrainableWeightsConsistency();let u=null!=(a=e).collectedTrainableWeights?si(a.collectedTrainableWeights):si(a.trainableWeights),h=si(e.nonTrainableWeights);n(`Total params: ${u+h}`),n(`Trainable params: ${u}`),n(`Non-trainable params: ${h}`),n("_".repeat(t))}(this,e,t,r)}compile(e){if(null==e.loss&&(e.loss=[]),this.loss=e.loss,"string"==typeof e.optimizer)this.optimizer_=function(e){let t={Adagrad:()=>ii.train.adagrad(.01),Adadelta:()=>ii.train.adadelta(1,.95,aN()),Adam:()=>ii.train.adam(.001,.9,.999,aN()),Adamax:()=>ii.train.adamax(.002,.9,.999,aN(),0),RMSProp:()=>ii.train.rmsprop(.001,.9,0,aN()),SGD:()=>ii.train.sgd(.01)};if(t.adagrad=t.Adagrad,t.adadelta=t.Adadelta,t.adam=t.Adam,t.adamax=t.Adamax,t.rmsprop=t.RMSProp,t.sgd=t.SGD,e in t)return t[e]();throw new nF(`Unknown Optimizer ${e}`)}(e.optimizer),this.isOptimizerOwned=!0;else{if(!(e.optimizer instanceof sX.Optimizer))throw new nF("User-defined optimizer must be an instance of tf.Optimizer.");this.optimizer_=e.optimizer,this.isOptimizerOwned=!1}let t=[];if(Array.isArray(e.loss)||"string"==typeof e.loss||"function"==typeof e.loss)if(Array.isArray(e.loss)){if(e.loss.length!==this.outputs.length)throw new nF(`When passing an Array as loss, it should have one entry per model output. The model has ${this.outputs.length} output(s), but you passed loss=${e.loss}.`);t=e.loss.map(e=>s6(e))}else{let r=s6(e.loss);this.outputs.forEach(e=>{t.push(r)})}else{for(let t in e.loss=e.loss,e.loss)if(-1===this.outputNames.indexOf(t))throw new nF(`Unknown entry in loss dictionary: "${t}". Only expected the following keys: ${this.outputNames}`);for(let r of this.outputNames)null==e.loss[r]&&console.warn(`Output "${r}" is missing from loss dictionary. We assume this was done on purpose, and we will not be expecting data to be passed to ${r} during training`),t.push(s6(e.loss[r]))}this.lossFunctions=t,this.feedOutputNames=[],this.feedOutputShapes=[],this.feedLossFns=[];for(let e=0;e<this.outputs.length;++e){let t=this.internalOutputShapes[e],r=this.outputNames[e];this.feedOutputNames.push(r),this.feedOutputShapes.push(t),this.feedLossFns.push(this.lossFunctions[e])}let r=[];this.metrics=e.metrics,this.metricsNames=["loss"],this.metricsTensors=[],ar("loss",()=>{for(let e=0;e<this.outputs.length;++e){if(-1!==r.indexOf(e))continue;let t=this.lossFunctions[e];this.outputs.length>1&&(this.metricsTensors.push([t,e]),this.metricsNames.push(this.outputNames[e]+"_loss"))}});let n=function(e,t){let r;if(null==e||Array.isArray(e)&&0===e.length)return t.map(e=>[]);if("string"==typeof e||"function"==typeof e)r=[e];else if(Array.isArray(e)||"object"==typeof e)r=e;else throw TypeError(`Type of metrics argument not understood. Expected an string,function, Array, or Object, found: ${e}`);if(Array.isArray(r))return t.map(e=>r);{let e=[];for(let n of t){let t=r.hasOwnProperty(n)?r[n]:[];Array.isArray(t)||(t=[t]),e.push(t)}return e}}(e.metrics,this.outputNames),a=(e,t,r)=>{this.outputNames.length>1&&(t=this.outputNames[e]+"_"+t),this.metricsNames.push(t),this.metricsTensors.push([r,e])};ar("metric",()=>{for(let e=0;e<this.outputs.length;++e){if(-1===r.indexOf(e))(t=>{let r,n,s;for(let i of t){let t;if("string"==typeof i&&-1!==["accuracy","acc","crossentropy","ce"].indexOf(i)){let t,a=this.internalOutputShapes[e];1===a[a.length-1]||this.lossFunctions[e]===s3?-1!==["accuracy","acc"].indexOf(i)?n=s8:-1!==["crossentropy","ce"].indexOf(i)&&(n=it):this.lossFunctions[e]===s2?-1!==["accuracy","acc"].indexOf(i)?n=ir:-1!==["crossentropy","ce"].indexOf(i)&&(n=s2):-1!==["accuracy","acc"].indexOf(i)?n=s7:-1!==["crossentropy","ce"].indexOf(i)&&(n=s1),-1!==["accuracy","acc"].indexOf(i)?t="acc":-1!==["crossentropy","ce"].indexOf(i)&&(t="ce"),s=n,r=""+t}else s=function(e){if("string"==typeof e&&e in ia)return ia[e];if("string"!=typeof e&&null!=e)return e;throw new nF(`Unknown metric ${e}`)}(i),r=""+is(i);ar(r,()=>{t=s}),a(e,r,t)}})(n[e])}}),this.collectedTrainableWeights=this.trainableWeights}checkTrainableWeightsConsistency(){null!=this.collectedTrainableWeights&&this.trainableWeights.length!==this.collectedTrainableWeights.length&&console.warn("Discrepancy between trainableweights and collected trainable weights. Did you set `model.trainable` without calling `model.compile()` afterwards?")}evaluate(e,t,r={}){let n=null==r.batchSize?32:r.batchSize;iC(n);let a=this.standardizeUserDataXY(e,t,!0,n);try{let e=a[0].concat(a[1]);this.makeTestFunction();let t=this.testFunction,s=this.testLoop(t,e,n,r.verbose,r.steps);return nP(s)}finally{iN(a[0],e),iN(a[1],t)}}async evaluateDataset(e,t){return this.makeTestFunction(),iw(this,e,t)}checkNumSamples(e,t,r,n="steps"){let a;if(null!=r){if(a=null,null!=t)throw new nF(`If ${n} is set, batchSize must be null or undefined.Got batchSize = ${t}`)}else if(null!=e)a=Array.isArray(e)?e[0].shape[0]:e.shape[0];else throw new nF(`Either the input data should have a defined shape, or ${n} shoud be specified.`);return a}execute(e,t){if(Array.isArray(t)&&0===t.length)throw new nF("`outputs` is an empty Array, which is not allowed.");let r=Array.isArray(t),n=r?t:[t],a=this.retrieveSymbolicTensors(n),s=new sv;if(e instanceof rm.Tensor&&(e=[e]),Array.isArray(e)){if(e.length!==this.inputs.length)throw new nF(`The number of inputs provided (${e.length}) does not match the number of inputs of this model (${this.inputs.length}).`);for(let t=0;t<this.inputs.length;++t)s.add(this.inputs[t],e[t])}else for(let t of this.inputs){let r=e[t.name];if(null==r)throw new nF(`No value is provided for the model's input ${t.name}`);s.add(t,r)}let i=sI(a,s);return r?i:i[0]}retrieveSymbolicTensors(e){let t=nz(null,e.length),r=e.length;for(let n of this.layers){let a=Array.isArray(n.output)?n.output:[n.output],s=a.map(e=>e.name);for(let n=0;n<e.length;++n){let i=s.indexOf(e[n]);if(-1!==i&&(t[n]=a[i],r--),0===r)break}if(0===r)break}if(r>0){let r=[];throw t.forEach((t,n)=>{null==t&&r.push(e[n])}),new nF(`Cannot find SymbolicTensors for output name(s): ${JSON.stringify(r)}`)}return t}predictLoop(e,t=32,r=!1){return n$.tidy(()=>{let n=this.checkNumSamples(e);if(r)throw new nD("Verbose predictLoop() is not implemented yet.");let a=iS(n,t),s=this.outputs.map(e=>[]);for(let t=0;t<a.length;++t)n$.tidy(()=>{let r=iI(e,a[t][0],a[t][1]),n=[];if(Array.isArray(r))for(let e=0;e<r.length;++e)n.push({key:this.inputs[e],value:r[e]});else n.push({key:this.inputs[0],value:r});let s=new sv(n);return sI(this.outputs,s)}).forEach((e,t)=>s[t].push(e));return nP(s.map(e=>t5.concat(e,0)))})}predict(e,t={}){let r=iT(e);iE(r,this.inputNames,this.feedInputShapes,!1);try{let e=null==t.batchSize?32:t.batchSize;return iC(e),this.predictLoop(r,e)}finally{iN(r,e)}}predictOnBatch(e){iE(e,this.inputNames,this.feedInputShapes,!0);let t=(Array.isArray(e)?e[0]:e).shape[0];return this.predictLoop(e,t)}standardizeUserDataXY(e,t,r=!0,n){if(null==this.optimizer_)throw new nE("You must compile a model before training/testing. Use LayersModel.compile(modelCompileArgs).");let a=[];for(let e=0;e<this.feedOutputShapes.length;++e){let t=this.feedOutputShapes[e];this.feedLossFns[e]===s2?a.push(t.slice(0,t.length-1).concat([1])):a.push(t)}e=iA(e,this.feedInputNames,this.feedInputShapes,!1,"input"),t=iA(t,this.feedOutputNames,a,!1,"target");var s=e,i=t;let o=nj(s.map(e=>e.shape[0]));o.sort();let l=nj(i.map(e=>e.shape[0]));if(l.sort(),o.length>1)throw new nF(`All input Tensors (x) should have the same number of samples. Got array shapes: ${JSON.stringify(s.map(e=>e.shape))}`);if(l.length>1)throw new nF(`All target Tensors (y) should have the same number of samples. Got array shapes: ${JSON.stringify(i.map(e=>e.shape))}`);if(o.length>0&&l.length>0&&!nR.util.arraysEqual(o,l))throw new nF(`Input Tensors should have the same number of samples as target Tensors. Found ${o[0]} input sample(s) and ${l[0]} target sample(s).`);if(!function(e,t,r){let n=[sJ,s3,s1];for(let a=0;a<e.length;++a){let s=e[a],i=t[a],o=r[a];if(null!=i){if(i===s1&&1===s.shape[s.shape.length-1])throw new nF(`You are passing a target array of shape ${s.shape} while using a loss 'categorical_crossentropy'. 'categorical_crossentropy'expects targets to be binary matrices (1s and 0s) of shape [samples, classes].`);if(-1!==n.indexOf(i)){let e=s.shape.slice(1),t=o.slice(1);for(let r=0;r<e.length;++r){let n=e[r],a=t[r];if(null!=a&&n!==a)throw new nF(`A target Tensor with shape ${s.shape} was passed for an output of shape ${o}, while using a loss function that expects targets to have the same shape as the output.`)}}}}}(t,this.feedLossFns,this.feedOutputShapes),this.stateful&&null!=n&&n>0&&e[0].shape[0]%n!=0)throw new nF(`In a stateful network, you should only pass inputs with a number of samples that is divisible by the batch size ${n}. Found: ${e[0].shape[0]} sample(s).`);return[e,t]}async standardizeUserData(e,t,r,n,a=!0,s){let[i,o]=this.standardizeUserDataXY(e,t,a,s);if(null!=r)throw Error("sample weight is not supported yet.");let l=null;if(null!=n){let e=im(n,this.outputNames);l=[];for(let t=0;t<e.length;++t)l.push(await ig(o[t],null,e[t]))}return[i,o,l]}testLoop(e,t,r,n=0,a){return n$.tidy(()=>{let s=this.checkNumSamples(t,r,a,"steps"),i=[];if(n>0)throw new nD("Verbose mode is not implemented yet.");if(null!=a)throw new nD("steps mode in testLoop() is not implemented yet");{let n=iS(s,r),a=(0,aC.tensor1d)(aT(0,s));for(let r=0;r<n.length;++r){let s=n[r][0],o=n[r][1],l=e(ik(t,aE(a,s,o-s)));if(0===r)for(let e=0;e<l.length;++e)i.push((0,k.scalar)(0));for(let e=0;e<l.length;++e){let t=l[e];i[e]=P.add(i[e],b.mul(o-s,t))}}for(let e=0;e<i.length;++e)i[e]=C.div(i[e],s)}return i})}getDedupedMetricsNames(){let e=this.metricsNames,t=[];for(let r=0;r<e.length;++r){let n=e[r],a=n;if(nM(e,n)>1){let t=nM(e.slice(0,r),n);a+=`_${t}`}t.push(a)}return t}makeTrainFunction(){return e=>{let t=[],r=e.slice(0,this.inputs.length),n=e.slice(this.inputs.length,this.inputs.length+this.outputs.length),a=e.slice(this.inputs.length+this.outputs.length,this.inputs.length+2*this.outputs.length),s=[],i=()=>{let e,i=[];for(let e=0;e<this.inputs.length;++e)i.push({key:this.inputs[e],value:r[e]});let o=new sv(i),l=sI(this.outputs,o,{training:!0});for(let r=0;r<this.lossFunctions.length;++r){var u,h;let s=(0,this.lossFunctions[r])(n[r],l[r]);null!=a[r]&&(u=s,h=a[r],s=(0,b.mul)(u,h));let i=r6.mean(s);t.push(i),e=0===r?s:P.add(e,s)}for(let e=0;e<this.metricsTensors.length;++e){let r;if(this.outputs.length>1&&e<this.outputs.length)r=t[e];else{let t=this.metricsTensors[e][0],a=this.metricsTensors[e][1];r=r6.mean(t(n[a],l[a]))}n$.keep(r),s.push(r)}return e=r6.mean(e),this.calculateLosses().forEach(t=>{e=P.add(e,t)}),e},o=this.collectedTrainableWeights.map(e=>e.read());return[this.optimizer_.minimize(i,!0,o)].concat(s)}}makeTestFunction(){this.testFunction=e=>n$.tidy(()=>{let t,r=[],n=e.slice(0,this.inputs.length),a=e.slice(this.inputs.length,this.inputs.length+this.outputs.length),s=[];for(let e=0;e<this.inputs.length;++e)s.push({key:this.inputs[e],value:n[e]});let i=new sv(s),o=sI(this.outputs,i);for(let e=0;e<this.lossFunctions.length;++e){let n=this.lossFunctions[e],s=r6.mean(n(a[e],o[e]));t=0===e?s:P.add(t,s),r.push(t)}for(let e=0;e<this.metricsTensors.length;++e){let t=this.metricsTensors[e][0],n=this.metricsTensors[e][1],s=r6.mean(t(a[n],o[n]));r.push(s)}return r})}async fit(e,t,r={}){let n,a,s,i,o,l,u,h,d;if(this.isTraining)throw Error("Cannot start training because another fit() call is ongoing.");this.isTraining=!0;try{let p,c,f,m=null==r.batchSize?32:r.batchSize;iC(m);let g=await this.standardizeUserData(e,t,r.sampleWeight,r.classWeight,!1,m);n=g[0],a=g[1],d=g[2];let x=!1;if(null!=r.validationData&&r.validationData.length>0){if(x=!0,2===r.validationData.length)o=r.validationData[0],l=r.validationData[1];else if(3===r.validationData.length)throw new nD("validationData including sample weights is not supported yet.");else throw new nF(`When passing validation data, it must contain 2 (valX, valY) or 3 (valX, valY, valSampleWeight) items; ${r.validationData} is invalid.`);let e=await this.standardizeUserData(o,l,null,null,!0,m);u=e[0],h=e[1],p=u.concat(h)}else if(null!=r.validationSplit&&r.validationSplit>0&&r.validationSplit<1){x=!0;let e=Math.floor(n[0].shape[0]*(1-r.validationSplit)),t=n[0].shape[0];u=iI(n,e,t),s=n,n=iI(n,0,e),h=iI(a,e,t),i=a,a=iI(a,0,e),p=u.concat(h)}else null!=r.validationSteps&&(x=!0);let y=n.concat(a).concat(d);this.checkTrainableWeightsConsistency();let b=this.makeTrainFunction(),v=this.getDedupedMetricsNames();x?(this.makeTestFunction(),c=this.testFunction,f=v.slice().concat(v.map(e=>"val_"+e))):(c=null,p=[],f=v.slice());let w=sV(r.callbacks,r.yieldEvery);return await this.fitLoop(b,y,v,m,r.epochs,r.verbose,w,c,p,r.shuffle,f,r.initialEpoch,null,null)}finally{this.isTraining=!1,iN(n,e),iN(a,t),iN(s,e),iN(i,t),iN(u,o),iN(h,l),null!=d&&n$.dispose(d)}}async fitLoop(e,t,r,n,a,s,i,o,l,u,h,d,p,c){let f;null==n&&(n=32),null==a&&(a=1),null==u&&(u=!0),null==d&&(d=0);let m=!1;if(null!=o&&null!=l&&(m=!0),null!=c&&(m=!0,null==p))throw new nF("Can only use `validationSteps` when doing step-wise training, i.e., `stepsPerEpoch` must be set.");let g=this.checkNumSamples(t,n,p,"steps_per_epoch");null!=g&&(f=aT(0,g)),null==s&&(s=1);let{callbackList:x,history:y}=sq(i,s,a,d,g,p,n,m,h);x.setModel(this),this.history=y,await x.onTrainBegin(),this.stopTraining_=!1;for(let s=d;s<a;++s){await x.onEpochBegin(s);let a={};if(null!=p)throw new nD("stepsPerEpoch mode is not implemented yet.");{if("batch"===u)throw new nD("batch shuffling is not implemneted yet");u&&nR.util.shuffle(f);let s=(0,aC.tensor1d)(f),i=iS(g,n);for(let u=0;u<i.length;++u){let h={};if(await x.onBatchBegin(u,h),n$.tidy(()=>{let d=i[u][0],p=i[u][1],c=aE(s,d,p-d);h.batch=u,h.size=p-d;let f=e(ik(t,c));for(let e=0;e<r.length;++e){let t=r[e],n=f[e];h[t]=n,n$.keep(n)}if(u===i.length-1&&m){let e=this.testLoop(o,l,n);for(let t=0;t<r.length;++t){let n=r[t],s=e[t];n$.keep(s),a["val_"+n]=s}}}),await x.onBatchEnd(u,h),sM(h),this.stopTraining_)break}s.dispose()}if(await x.onEpochEnd(s,a),this.stopTraining_)break}return await x.onTrainEnd(),await this.history.syncData(),this.history}async fitDataset(e,t){return ib(this,e,t)}async trainOnBatch(e,t){let r=await this.standardizeUserData(e,t),n=r[0],a=r[1],s=this.makeTrainFunction()(n.concat(a)),i=[];for(let e of s){let t=await e.data();i.push(t[0])}return n$.dispose(s),iN(r[0],e),iN(r[1],t),nP(i)}getNamedWeights(e){let t=[],r=null!=e&&e.trainableOnly,n=r?this.trainableWeights:this.weights,a=this.getWeights(r);for(let e=0;e<n.length;++e)(!r||n[e].trainable)&&t.push({name:n[e].originalName,tensor:a[e]});return t}set stopTraining(e){this.stopTraining_=e}get stopTraining(){return this.stopTraining_}get optimizer(){return this.optimizer_}set optimizer(e){this.optimizer_!==e&&(this.optimizer_=e,this.isOptimizerOwned=!1)}dispose(){let e=super.dispose();if(0===e.refCountAfterDispose&&null!=this.optimizer&&this.isOptimizerOwned){let t=n$.memory().numTensors;this.optimizer_.dispose(),e.numDisposedVariables+=t-n$.memory().numTensors}return e}getLossIdentifiers(){let e;if("string"==typeof this.loss)e=nW(this.loss);else if(Array.isArray(this.loss)){for(let e of this.loss)if("string"!=typeof e)throw Error("Serialization of non-string loss is not supported.");e=this.loss.map(e=>nW(e))}else{let t=Object.keys(this.loss);e={};let r=this.loss;for(let n of t)if("string"==typeof r[n])e[n]=nW(r[n]);else throw Error("Serialization of non-string loss is not supported.")}return e}getMetricIdentifiers(){if("string"==typeof this.metrics||"function"==typeof this.metrics)return[nW(is(this.metrics))];{if(Array.isArray(this.metrics))return this.metrics.map(e=>nW(is(e)));let e={};for(let t in this.metrics)e[t]=nW(is(this.metrics[t]));return e}}getTrainingConfig(){return{loss:this.getLossIdentifiers(),metrics:this.getMetricIdentifiers(),optimizer_config:{class_name:this.optimizer.getClassName(),config:this.optimizer.getConfig()}}}loadTrainingConfig(e){let t,r;if(null!=e.weighted_metrics)throw Error("Loading weight_metrics is not supported yet.");if(null!=e.loss_weights)throw Error("Loading loss_weights is not supported yet.");if(null!=e.sample_weight_mode)throw Error("Loading sample_weight_mode is not supported yet.");let n=sK(ih(e.optimizer_config));if("string"==typeof e.loss)t=nG(e.loss);else if(Array.isArray(e.loss))t=e.loss.map(e=>nG(e));else if(null!=e.loss)for(let r in t={},e.loss)t[r]=nG(e.loss[r]);if(Array.isArray(e.metrics))r=e.metrics.map(e=>nG(e));else if(null!=e.metrics)for(let t in r={},e.metrics)r[t]=nG(e.metrics[t]);this.compile({loss:t,metrics:r,optimizer:n})}async save(e,t){if("string"==typeof e){let t=sj.io.getSaveHandlers(e);if(0===t.length)throw new nF(`Cannot find any save handlers for URL '${e}'`);if(t.length>1)throw new nF(`Found more than one (${t.length}) save handlers for URL '${e}'`);e=t[0]}if(null==e.save)throw new nF("LayersModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");let r=await sj.io.encodeWeights(this.getNamedWeights(t)),n={modelTopology:this.toJSON(null,!1),format:"layers-model",generatedBy:`TensorFlow.js tfjs-layers v${id}`,convertedBy:null};if(null!=t&&t.includeOptimizer&&null!=this.optimizer){n.trainingConfig=this.getTrainingConfig();let{data:e,specs:t}=await sj.io.encodeWeights(await this.optimizer.getWeights(),"optimizer");r.specs.push(...t),r.data=sj.io.concatenateArrayBuffers([r.data,e])}return null!=this.userDefinedMetadata&&(io(this.userDefinedMetadata,this.name,!0),n.userDefinedMetadata=this.userDefinedMetadata),n.weightData=r.data,n.weightSpecs=r.specs,e.save(n)}setUserDefinedMetadata(e){io(e,this.name),this.userDefinedMetadata=e}getUserDefinedMetadata(){return this.userDefinedMetadata}}iF.className="Model",nJ.registerClass(iF);class iD extends iF{}iD.className="Functional",nJ.registerClass(iD);var nJ=nJ;async function iO(e,t){"modelTopology"in e||(e={modelTopology:e});let r=e.modelTopology;null!=r.model_config&&(r=r.model_config);let n=sK(ih(r),t);if(null!=e.weightsManifest){let t=await sj.io.loadWeights(e.weightsManifest,e.pathPrefix,n.weights.map(e=>e.originalName)),r={};for(let e of n.weights)r[e.originalName]=t[e.originalName];n.loadWeights(r),(0,n$.dispose)(t)}return n}async function iL(e,t){if(null==t&&(t={}),"string"==typeof e){let r=sj.io.getLoadHandlers(e,t);if(0===r.length)r.push(sj.io.browserHTTPRequest(e,t));else if(r.length>1)throw new nF(`Found more than one (${r.length}) load handlers for URL '${e}'`);e=r[0]}return iz(e,void 0,t)}async function iz(e,t,r){if(null==r&&(r={}),null==e.load)throw new nF("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");let n=await e.load(),a=n.modelTopology;null!=a.model_config&&(a=a.model_config);let s=null==r.strict||r.strict,i=null!=n.weightData&&null!=n.weightSpecs&&s,o=sK(ih(a),t,i),l=n.trainingConfig;if(null!=l&&o.loadTrainingConfig(l),null!=n.userDefinedMetadata&&o.setUserDefinedMetadata(n.userDefinedMetadata),null!=n.weightData){var u,h;let e,t,r;if(null==n.weightSpecs)throw new nF("LayersModel artifacts contains weight data, but not weight specs. Therefore loading of weights cannot proceed.");let{modelWeights:a,optimizerWeights:i}=(u=n.weightData,h=n.weightSpecs,e=sj.io.decodeWeights(u,h),t={},r=[],h.forEach(n=>{"optimizer"===n.group?r.push({name:n.name,tensor:e[n.name]}):t[n.name]=e[n.name]}),{modelWeights:t,optimizerWeights:r});o.loadWeights(a,s),null!=o.optimizer&&i.length>0&&await o.optimizer.setWeights(i),(0,n$.dispose)(a),(0,n$.dispose)(i.map(e=>e.tensor))}return o}class i_ extends iF{constructor(e){if(super({inputs:[],outputs:[]}),e=e||{},this.trainable=!0,this.built=!1,this.name=null!=e.name?e.name:n1("sequential_"),null!=e.layers)for(const t of e.layers)this.add(t)}checkShape(e){if(e.inboundNodes[0].outputTensors[0].shape.some(e=>e<0))throw new nF(`Negative dimension size caused by adding layer ${e.name} with input shape [${e.inboundNodes[0].inputTensors[0].shape}]`)}add(e){let t,r=e instanceof i_||e instanceof iF;if(r){if(1!==(t=e).outputs.length)throw new nF("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");if(1!==t.inputs.length)throw new nF("All layers in a Sequential model should have a single input tensor. For multi-input layers, use the functional API.")}if(0===this.outputs.length){if(0===e.inboundNodes.length){if(null==e.batchInputShape)throw new nF("The first layer in a Sequential model must get an `inputShape` or `batchInputShape` argument.");let t=sb({batchShape:e.batchInputShape,dtype:e.dtype,name:e.name+"_input"});e.apply(t)}if(r)this.outputs=t.outputs,this.inputs=t.inputs;else{if(1!==e.inboundNodes.length)throw new nF(`A layer added to a Sequential model must not already be connected somewhere else. LayersModel received layer ${e.name} which has ${e.inboundNodes.length} pre-existing inbound connections.`);if(1!==e.inboundNodes[0].outputTensors.length)throw new nF("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");this.checkShape(e),this.outputs=[e.inboundNodes[0].outputTensors[0]],this.inputs=function e(t,r,n){if((null==r||null!=n&&n>0)&&(r=t.sourceLayer,n=t.nodeIndex),0===r.inboundNodes.length)return[t];{let t=r.inboundNodes[n];if(0===t.inboundLayers.length)return t.inputTensors;{let r=[];for(let n=0;n<t.inboundLayers.length;n++)for(let a of e(t.inputTensors[n],t.inboundLayers[n],t.nodeIndices[n]))-1===r.indexOf(a)&&r.push(a);return r}}}(this.outputs[0])}this.inboundNodes=[],new sm({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:nz(null,this.inputs.length),outputMasks:[null],inputShapes:this.inputs.map(e=>e.shape),outputShapes:this.outputs[0].shape})}else{let t=e.apply(this.outputs[0]);if(Array.isArray(t))throw TypeError("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");this.checkShape(e),this.outputs=[t],this.inboundNodes[0].outputTensors=this.outputs,this.inboundNodes[0].outputShapes=[this.outputs[0].shape]}this.layers.push(e),this.built=!1}pop(){if(0===this.layers.length)throw TypeError("There are no layers in the model.");if(this.layers.pop(),0===this.layers.length)this.outputs=[],this.inboundNodes=[],this.outboundNodes=[];else{let e=this.layers.length-1;this.layers[e].outboundNodes=[],this.outputs=[this.layers[e].output],this.inboundNodes[0].outputTensors=this.outputs,this.inboundNodes[0].outputShapes=[this.outputs[0].shape]}}call(e,t){return null==this.model&&this.build(),this.model.call(e,t)}build(e){if(ss(e),0===this.inputs.length||0===this.outputs.length)throw TypeError("Sequential model cannot be built: model is empty. Add some layers first.");this.model=new iF({inputs:this.inputs,outputs:this.outputs[0],name:this.name+"_model"}),this.model.trainable=this.trainable,this.supportsMasking=this.model.supportsMasking,this.inputLayers=this.model.inputLayers,this.inputLayersNodeIndices=this.model.inputLayersNodeIndices,this.inputLayersTensorIndices=this.model.inputLayersTensorIndices,this.outputLayers=this.model.outputLayers,this.outputLayersNodeIndices=this.model.outputLayersNodeIndices,this.outputLayersTensorIndices=this.model.outputLayersTensorIndices,this.nodesByDepth=this.model.nodesByDepth,this.containerNodes=this.model.containerNodes,this.outputNames=this.model.outputNames,this.inputNames=this.model.inputNames,this.built=!0}countParams(){return this.built||this.build(),super.countParams()}summary(e,t,r=console.log){this.built||this.build(),super.summary(e,t,r)}setWeights(e){null==this.model&&this.build(),this.model.setWeights(e)}evaluate(e,t,r={}){if(!this.built)throw new nE("The model needs to be compiled before being used.");return this.model.evaluate(e,t,r)}async evaluateDataset(e,t){if(!this.built)throw new nE("The model needs to be compiled before being used.");return this.model.evaluateDataset(e,t)}predict(e,t={}){return null==this.model&&this.build(),this.model.predict(e,t)}predictOnBatch(e){return null==this.model&&this.build(),this.model.predictOnBatch(e)}compile(e){this.build(),this.model.compile(e),this.optimizer_=this.model.optimizer,this.isOptimizerOwned=this.model.isOptimizerOwned,this.loss=this.model.loss,this.metrics=this.model.metrics,this.metricsTensors=this.model.metricsTensors,this.metricsNames=this.model.metricsNames}get optimizer(){return null==this.model?void 0:this.model.optimizer}set optimizer(e){this.model.optimizer=e}async fit(e,t,r={}){if(!this.built)throw new nE("The model needs to be compiled before being used.");return this.model.fit(e,t,r)}async fitDataset(e,t){if(!this.built)throw new nE("The model needs to be compiled before being used.");return this.model.fitDataset(e,t)}async trainOnBatch(e,t){return this.model.trainOnBatch(e,t)}static fromConfig(e,t,r={},n=!1){let a,s={};if(t instanceof Array){if(null==t[0].className||"Merge"===t[0].className)throw new nF("Legacy serialization format not supported yet.");a=t}else nR.util.assert(null!=t.layers,()=>"When the config data for a Sequential model is not an Array, it must be an Object that contains the 'layers' field."),a=t.layers,delete t.layers,s=t;let i=new e(s);if(!(i instanceof i_))throw new nD(`Sequential.fromConfig called on non-Sequential input: ${i}`);for(let e of a){let t=sK(e,void 0,n);n&&t.setFastWeightInitDuringBuild(!0),i.add(t)}return i}set stopTraining(e){if(null==this.model)throw new nF("Cannot set the stopTraining property of a sequential model before it is compiled.");this.model.stopTraining=e}get stopTraining(){if(null==this.model)throw new nF("Cannot get the stopTraining property of a sequential model before it is compiled.");return this.model.stopTraining}getConfig(){let e=[];for(let t of this.layers){let r={};r.className=t.getClassName(),r.config=t.getConfig(),e.push(r)}return{name:this.name,layers:e}}}function iM(e){return new iF(e)}function iP(e){return new i_(e)}function iB(e){return sb(e)}function iW(e,t){sH.registerCallbackConstructor(e,t)}i_.className="Sequential",nJ.registerClass(i_);var nJ=nJ,nJ=nJ;class iG extends nJ.Serializable{getConfig(){return{}}}class iU extends iG{apply(e,t=1){return function(e,t=1){if(1!==t)throw new nD(`Support for alpha values other than 1 (${t}) is not implemented yet.`);return rP.elu(e)}(e,t)}}iU.className="elu",nJ.registerClass(iU);class iV extends iG{apply(e){return nm.selu(e)}}iV.className="selu",nJ.registerClass(iV);class iH extends iG{apply(e){return nu.relu(e)}}iH.className="relu",nJ.registerClass(iH);class iq extends iG{apply(e){return(0,n$.tidy)(()=>r7.minimum(6,nu.relu(e)))}}iq.className="relu6",nJ.registerClass(iq);class ij extends iG{apply(e){return e}}ij.className="linear",nJ.registerClass(ij);class iX extends iG{apply(e){return t1.sigmoid(e)}}iX.className="sigmoid",nJ.registerClass(iX);class iK extends iG{apply(e){return(0,n$.tidy)(()=>{let t=P.add(.5,b.mul(.2,e));return rE.clipByValue(t,0,1)})}}iK.className="hardSigmoid",nJ.registerClass(iK);class iY extends iG{apply(e){return nb.softplus(e)}}iY.className="softplus",nJ.registerClass(iY);class iZ extends iG{apply(e){return(0,n$.tidy)(()=>C.div(e,P.add(rf.abs(e),1)))}}iZ.className="softsign",nJ.registerClass(iZ);class iJ extends iG{apply(e){return nk.tanh(e)}}iJ.className="tanh",nJ.registerClass(iJ);class iQ extends iG{apply(e,t=-1){return ny.softmax(e,t)}}iQ.className="softmax",nJ.registerClass(iQ);class i0 extends iG{apply(e,t=-1){return rQ.logSoftmax(e,t)}}i0.className="logSoftmax",nJ.registerClass(i0);class i1 extends iG{apply(e){return(0,n$.tidy)(()=>n$.tidy(()=>{let t=Math.sqrt(2),r=b.mul(.5,P.add(1,rB.erf(C.div(e,t))));return b.mul(e,r)}))}}i1.className="gelu",nJ.registerClass(i1);class i2 extends iG{apply(e){return(0,n$.tidy)(()=>b.mul(.5,b.mul(e,P.add(1,nk.tanh(b.mul(S.sqrt(C.div(2,Math.PI)),P.add(e,b.mul(.044715,tk.pow(e,3)))))))))}}i2.className="gelu_new",nJ.registerClass(i2);class i3 extends iG{apply(e){return(0,n$.tidy)(()=>b.mul(e,nk.tanh(nb.softplus(e))))}}i3.className="mish",nJ.registerClass(i3);class i4 extends iG{apply(e,t=1){return(0,n$.tidy)(()=>b.mul(t1.sigmoid(b.mul(e,t)),e))}}function i5(e){return e.getClassName()}function i6(e,t={}){return nH(e,nJ.SerializationMap.getMap().classNameMap,t,"activation")}function i8(e){if(null==e){let e={};return e.className="linear",e.config={},i6(e)}if("string"==typeof e){let t={};return t.className=e,t.config={},i6(t)}return e instanceof iG?e:i6(e)}i4.className="swish",nJ.registerClass(i4);var nJ=nJ;function i7(e){if(null!=e&&"object"!=typeof e)throw Error(`Argument to L1L2 regularizer's constructor is expected to be an object, but received: ${e}`)}class i9 extends nJ.Serializable{}class oe extends i9{constructor(e){super(),i7(e),this.l1=null==e||null==e.l1?.01:e.l1,this.l2=null==e||null==e.l2?.01:e.l2,this.hasL1=0!==this.l1,this.hasL2=0!==this.l2}apply(e){return(0,n$.tidy)(()=>{let t=(0,tx.zeros)([1]);return this.hasL1&&(t=(0,P.add)(t,(0,F.sum)(b.mul(this.l1,(0,rf.abs)(e))))),this.hasL2&&(t=(0,P.add)(t,(0,F.sum)(b.mul(this.l2,aB(e))))),E.reshape(t,[])})}getConfig(){return{l1:this.l1,l2:this.l2}}static fromConfig(e,t){return new e({l1:t.l1,l2:t.l2})}}oe.className="L1L2",nJ.registerClass(oe);let ot={l1l2:"L1L2"};function or(e,t={}){return nH(e,nJ.SerializationMap.getMap().classNameMap,t,"regularizer")}function on(e){return null==e?null:"string"==typeof e?or({className:e in ot?ot[e]:e,config:{}}):e instanceof i9?e:or(e)}class oa extends sx{constructor(e){super(null==e?{}:e),this.supportsMasking=!0,null!=e&&(this.maxValue=e.maxValue)}call(e,t){e=sa(e);let r=(0,nu.relu)(e);return null!=this.maxValue&&(r=(0,rE.clipByValue)(r,0,this.maxValue)),r}computeOutputShape(e){return e}getConfig(){let e={maxValue:this.maxValue};return Object.assign(e,super.getConfig()),e}}oa.className="ReLU",nJ.registerClass(oa);class os extends sx{constructor(e){super(null==e?{}:e),this.DEFAULT_ALPHA=.3,null==e&&(e={}),this.alpha=null==e.alpha?this.DEFAULT_ALPHA:e.alpha}call(e,t){let r=sa(e);return(0,rY.leakyRelu)(r,this.alpha)}computeOutputShape(e){return e}getConfig(){let e={alpha:this.alpha};return Object.assign(e,super.getConfig()),e}}os.className="LeakyReLU",nJ.registerClass(os);class oi extends sx{constructor(e){if(super(null==e?{}:e),this.DEFAULT_ALPHA_INITIALIZER="zeros",null==e&&(e={}),this.supportsMasking=!0,this.alphaInitializer=st(e.alphaInitializer||this.DEFAULT_ALPHA_INITIALIZER),this.alphaRegularizer=on(e.alphaRegularizer),this.alphaConstraint=sD(e.alphaConstraint),null==e.sharedAxes)this.sharedAxes=null;else if(Array.isArray(e.sharedAxes))this.sharedAxes=e.sharedAxes;else if("number"==typeof e.sharedAxes)this.sharedAxes=[e.sharedAxes];else throw new nF(`Expected sharedAxes to be a number or an array of numbers, but got ${e.sharedAxes}`)}build(e){let t=(e=ss(e)).slice(1);if(null!=this.sharedAxes)for(let e of this.sharedAxes)t[e-1]=1;this.alpha=this.addWeight("alpha",t,"float32",this.alphaInitializer,this.alphaRegularizer,!0,this.alphaConstraint);let r={};if(null!=this.sharedAxes)for(let t=1;t<e.length;++t)r[t]=e[t];this.inputSpec=[new sp({ndim:e.length,axes:r})],this.built=!0}call(e,t){return e=sa(e),(0,ni.prelu)(e,this.alpha.read())}getConfig(){let e={alphaInitializer:nV(this.alphaInitializer),alphaRegularizer:nV(this.alphaRegularizer),alphaConstraint:nV(this.alphaConstraint),sharedAxes:this.sharedAxes};return Object.assign(e,super.getConfig()),e}}oi.className="PReLU",nJ.registerClass(oi);class oo extends sx{constructor(e){if(super(null==e?{}:e),this.DEFAULT_ALPHA=1,null==e&&(e={}),null!=e.alpha&&e.alpha!==this.DEFAULT_ALPHA)throw new nD(`Non-default alpha value (${e.alpha}) is not supported by the ELU layer yet.`);this.alpha=null==e.alpha?this.DEFAULT_ALPHA:e.alpha}call(e,t){let r=sa(e);return(0,rP.elu)(r)}computeOutputShape(e){return e}getConfig(){let e={alpha:this.alpha};return Object.assign(e,super.getConfig()),e}}oo.className="ELU",nJ.registerClass(oo);class ol extends sx{constructor(e){super(null==e?{}:e),this.DEFAULT_THETA=1,null==e&&(e={}),this.theta=null==e.theta?this.DEFAULT_THETA:e.theta}call(e,t){let r=sa(e);return(0,b.mul)(r,(0,y.cast)((0,e1.greater)(r,this.theta),"float32"))}computeOutputShape(e){return e}getConfig(){let e={theta:this.theta};return Object.assign(e,super.getConfig()),e}}ol.className="ThresholdedReLU",nJ.registerClass(ol);class ou extends sx{constructor(e){super(null==e?{}:e),this.DEFAULT_AXIS=1,null==e&&(e={}),this.softmax=new iQ().apply,this.axis=null==e.axis?this.DEFAULT_AXIS:e.axis}call(e,t){return(0,n$.tidy)(()=>{let r=sa(e),n=t.mask;if(null!=n){let e=(0,b.mul)((0,N.sub)((0,to.ones)(r.shape),(0,y.cast)(n,r.dtype)),(0,k.scalar)(-1e9));r=(0,P.add)(r,e)}if(this.axis instanceof Array)if(this.axis.length>1)return(0,eL.exp)((0,N.sub)(r,(0,r0.logSumExp)(r,this.axis,!0)));else return this.softmax(r,this.axis[0]);return this.softmax(r,this.axis)})}computeOutputShape(e){return e}getConfig(){let e={axis:this.axis};return Object.assign(e,super.getConfig()),e}}ou.className="Softmax",nJ.registerClass(ou);var oh=e.i(53415),av=av,od=e.i(27291),nJ=nJ;function op(e,t,r){if("number"==typeof e)return nz(e,t);if(e.length!==t)throw new nF(`The ${r} argument must be an integer or tuple of ${t} integers. Received: ${e.length} elements.`);for(let n=0;n<t;++n){let a=e[n];if(a!==parseInt(a.toString(),10))throw new nF(`The ${r} argument must be an integer or tuple of ${t} integers. Received: ${JSON.stringify(e)} including a non-integer number ${a}`)}return e}function oc(e,t,r,n,a=1){return null==e?e:Math.floor((("same"===r?e:e-(t+(t-1)*(a-1))+1)+n-1)/n)}function of(e,t,r,n){if(null==e)return null;if("valid"===n)e=e*t+aS([r-t,0]);else if("same"===n)e*=t;else throw new nF(`Unsupport padding mode: ${n}.`);return e}function om(e,t){return(0,n$.tidy)(()=>(n7(t),"channelsFirst"===t)?e$.transpose(e,[0,2,3,1]):e)}function og(e,t){return(0,n$.tidy)(()=>(n7(t),"channelsFirst"===t)?e$.transpose(e,[0,2,3,4,1]):e)}function ox(e,t,r,n=[1,1],a="valid",s,i,o=null){return(0,n$.tidy)(()=>{if(null==s&&(s=a$()),n7(s),3!==e.rank&&4!==e.rank)throw new nF(`conv2dWithBiasActivation expects input to be of rank 3 or 4, but received ${e.rank}.`);if(3!==t.rank&&4!==t.rank)throw new nF(`conv2dWithBiasActivation expects kernel to be of rank 3 or 4, but received ${e.rank}.`);let l=om(e,s);if("causal"===a)throw new nD("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");return l=av.conv2d({x:l,filter:t,strides:n,pad:"same"===a?"same":"valid",dilations:i,dataFormat:"NHWC",bias:r,activation:o}),"channelsFirst"===s&&(l=e$.transpose(l,[0,3,1,2])),l})}class oy extends sx{constructor(e,t){if(super(t),this.bias=null,this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_BIAS_INITIALIZER="zeros",oy.verifyArgs(t),this.rank=e,nY(this.rank,"rank"),1!==this.rank&&2!==this.rank&&3!==this.rank)throw new nD(`Convolution layer for rank other than 1, 2, or 3 (${this.rank}) is not implemented yet.`);if(this.kernelSize=op(t.kernelSize,e,"kernelSize"),this.strides=op(null==t.strides?1:t.strides,e,"strides"),this.padding=null==t.padding?"valid":t.padding,n9(this.padding),this.dataFormat=null==t.dataFormat?"channelsLast":t.dataFormat,n7(this.dataFormat),this.activation=i8(t.activation),this.useBias=null==t.useBias||t.useBias,this.biasInitializer=st(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.biasConstraint=sD(t.biasConstraint),this.biasRegularizer=on(t.biasRegularizer),this.activityRegularizer=on(t.activityRegularizer),this.dilationRate=op(null==t.dilationRate?1:t.dilationRate,e,"dilationRate"),1===this.rank&&Array.isArray(this.dilationRate)&&1!==this.dilationRate.length)throw new nF(`dilationRate must be a number or an array of a single number for 1D convolution, but received ${JSON.stringify(this.dilationRate)}`);if(2===this.rank){if("number"==typeof this.dilationRate)this.dilationRate=[this.dilationRate,this.dilationRate];else if(2!==this.dilationRate.length)throw new nF(`dilationRate must be a number or array of two numbers for 2D convolution, but received ${JSON.stringify(this.dilationRate)}`)}else if(3===this.rank){if("number"==typeof this.dilationRate)this.dilationRate=[this.dilationRate,this.dilationRate,this.dilationRate];else if(3!==this.dilationRate.length)throw new nF(`dilationRate must be a number or array of three numbers for 3D convolution, but received ${JSON.stringify(this.dilationRate)}`)}}static verifyArgs(e){if(n_("kernelSize"in e,"required key 'kernelSize' not in config"),"number"!=typeof e.kernelSize&&!nK(e.kernelSize,"number",1,3))throw new nF(`BaseConv expects config.kernelSize to be number or number[] with length 1, 2, or 3, but received ${JSON.stringify(e.kernelSize)}.`)}getConfig(){let e={kernelSize:this.kernelSize,strides:this.strides,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,activation:i5(this.activation),useBias:this.useBias,biasInitializer:nV(this.biasInitializer),biasRegularizer:nV(this.biasRegularizer),activityRegularizer:nV(this.activityRegularizer),biasConstraint:nV(this.biasConstraint)};return Object.assign(e,super.getConfig()),e}}class ob extends oy{constructor(e,t){super(e,t),this.kernel=null,ob.verifyArgs(t),this.filters=t.filters,nY(this.filters,"filters"),this.kernelInitializer=st(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.kernelConstraint=sD(t.kernelConstraint),this.kernelRegularizer=on(t.kernelRegularizer)}build(e){e=ss(e);let t="channelsFirst"===this.dataFormat?1:e.length-1;if(null==e[t])throw new nF(`The channel dimension of the input should be defined. Found ${e[t]}`);let r=e[t],n=this.kernelSize.concat([r,this.filters]);this.kernel=this.addWeight("kernel",n,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[{ndim:this.rank+2,axes:{[t]:r}}],this.built=!0}call(e,t){return(0,n$.tidy)(()=>{let t;e=sa(e);let r=null==this.bias?null:this.bias.read(),n=nZ(this.activation.getClassName());if(null!=n&&2===this.rank)t=ox(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate,n);else{if(1===this.rank)t=function(e,t,r,n=1,a="valid",s,i=1){return(0,n$.tidy)(()=>{if(null==s&&(s=a$()),n7(s),3!==e.shape.length)throw new nF(`The input of a conv1dWithBias operation should be 3, but is ${e.shape.length} instead.`);if(3!==t.shape.length)throw new nF(`The kernel for a conv1dWithBias operation should be 3, but is ${t.shape.length} instead`);if(null!=r&&1!==r.shape.length)throw new nF(`The bias for a conv1dWithBias operation should be 1, but is ${r.shape.length} instead`);if("channelsFirst"===s&&(e=e$.transpose(e,[0,2,1])),"causal"===a)throw new nD("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");let o=rF.conv1d(e,t,n,"same"===a?"same":"valid","NWC",i);return null!=r&&(o=aG(o,r)),o})}(e,this.kernel.read(),r,this.strides[0],this.padding,this.dataFormat,this.dilationRate[0]);else if(2===this.rank)t=ox(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate);else if(3===this.rank)t=function(e,t,r,n=[1,1,1],a="valid",s,i){return(0,n$.tidy)(()=>{if(null==s&&(s=a$()),n7(s),4!==e.rank&&5!==e.rank)throw new nF(`conv3dWithBias expects input to be of rank 4 or 5, but received ${e.rank}.`);if(4!==t.rank&&5!==t.rank)throw new nF(`conv3dWithBias expects kernel to be of rank 4 or 5, but received ${e.rank}.`);let o=og(e,s);if("causal"===a)throw new nD("The support for CAUSAL padding mode in conv3dWithBias is not implemented yet.");return o=od.conv3d(o,t,n,"same"===a?"same":"valid","NDHWC",i),null!=r&&(o=aG(o,r)),"channelsFirst"===s&&(o=e$.transpose(o,[0,4,1,2,3])),o})}(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate);else throw new nD("convolutions greater than 3D are not implemented yet.");null!=this.activation&&(t=this.activation.apply(t))}return t})}computeOutputShape(e){e=ss(e);let t=[],r="channelsLast"===this.dataFormat?e.slice(1,e.length-1):e.slice(2);for(let e=0;e<r.length;++e){let n=oc(r[e],this.kernelSize[e],this.padding,this.strides[e],"number"==typeof this.dilationRate?this.dilationRate:this.dilationRate[e]);t.push(n)}let n=[e[0]];return"channelsLast"===this.dataFormat?(n=n.concat(t)).push(this.filters):(n.push(this.filters),n=n.concat(t)),n}getConfig(){let e={filters:this.filters,kernelInitializer:nV(this.kernelInitializer),kernelRegularizer:nV(this.kernelRegularizer),kernelConstraint:nV(this.kernelConstraint)};return Object.assign(e,super.getConfig()),e}static verifyArgs(e){if(!("filters"in e)||"number"!=typeof e.filters||e.filters<1)throw new nF(`Convolution layer expected config.filters to be a 'number' > 0 but got ${JSON.stringify(e.filters)}`)}}class ov extends ob{constructor(e){super(2,e),ov.verifyArgs(e)}getConfig(){let e=super.getConfig();return delete e.rank,e}static verifyArgs(e){if("number"!=typeof e.kernelSize&&!nK(e.kernelSize,"number",1,2))throw new nF(`Conv2D expects config.kernelSize to be number or number[] with length 1 or 2, but received ${JSON.stringify(e.kernelSize)}.`)}}ov.className="Conv2D",nJ.registerClass(ov);class ow extends ob{constructor(e){super(3,e),ow.verifyArgs(e)}getConfig(){let e=super.getConfig();return delete e.rank,e}static verifyArgs(e){if("number"!=typeof e.kernelSize&&!(Array.isArray(e.kernelSize)&&(1===e.kernelSize.length||3===e.kernelSize.length)))throw new nF(`Conv3D expects config.kernelSize to be number or [number, number, number], but received ${JSON.stringify(e.kernelSize)}.`)}}ow.className="Conv3D",nJ.registerClass(ow);class oC extends ov{constructor(e){if(super(e),this.inputSpec=[new sp({ndim:4})],"same"!==this.padding&&"valid"!==this.padding)throw new nF(`Conv2DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(e){if(4!==(e=ss(e)).length)throw new nF("Input should have rank 4; Received input shape: "+JSON.stringify(e));let t="channelsFirst"===this.dataFormat?1:e.length-1;if(null==e[t])throw new nF("The channel dimension of the inputs should be defined. Found `None`.");let r=e[t],n=this.kernelSize.concat([this.filters,r]);this.kernel=this.addWeight("kernel",n,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new sp({ndim:4,axes:{[t]:r}})],this.built=!0}call(e,t){return n$.tidy(()=>{let t,r,n=sa(e);if(4!==n.shape.length)throw new nF(`Conv2DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${n.shape.length}`);let a=n.shape,s=a[0];"channelsFirst"===this.dataFormat?(t=2,r=3):(t=1,r=2);let i=a[t],o=a[r],l=this.kernelSize[0],u=this.kernelSize[1],h=this.strides[0],d=this.strides[1],p=[s,of(i,h,l,this.padding),of(o,d,u,this.padding),this.filters];"channelsLast"!==this.dataFormat&&(n=e$.transpose(n,[0,2,3,1]));let c=rD.conv2dTranspose(n,this.kernel.read(),p,this.strides,this.padding);return"channelsLast"!==this.dataFormat&&(c=e$.transpose(c,[0,3,1,2])),null!=this.bias&&(c=aG(c,this.bias.read(),this.dataFormat)),null!=this.activation&&(c=this.activation.apply(c)),c})}computeOutputShape(e){let t,r,n,a=(e=ss(e)).slice();"channelsFirst"===this.dataFormat?(t=1,r=2,n=3):(t=3,r=1,n=2);let s=this.kernelSize[0],i=this.kernelSize[1],o=this.strides[0],l=this.strides[1];return a[t]=this.filters,a[r]=of(a[r],o,s,this.padding),a[n]=of(a[n],l,i,this.padding),a}getConfig(){let e=super.getConfig();return delete e.dilationRate,e}}oC.className="Conv2DTranspose",nJ.registerClass(oC);class oI extends ow{constructor(e){if(super(e),this.inputSpec=[new sp({ndim:5})],"same"!==this.padding&&"valid"!==this.padding)throw new nF(`Conv3DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(e){if(5!==(e=ss(e)).length)throw new nF("Input should have rank 5; Received input shape: "+JSON.stringify(e));let t="channelsFirst"===this.dataFormat?1:e.length-1;if(null==e[t])throw new nF("The channel dimension of the inputs should be defined. Found `None`.");let r=e[t],n=this.kernelSize.concat([this.filters,r]);this.kernel=this.addWeight("kernel",n,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new sp({ndim:5,axes:{[t]:r}})],this.built=!0}call(e,t){return n$.tidy(()=>{let t,r,n,a=sa(e);if(5!==a.shape.length)throw new nF(`Conv3DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${a.shape.length}`);let s=a.shape,i=s[0];"channelsFirst"===this.dataFormat?(n=2,t=3,r=4):(n=1,t=2,r=3);let o=s[n],l=s[t],u=s[r],h=this.kernelSize[0],d=this.kernelSize[1],p=this.kernelSize[2],c=this.strides[0],f=this.strides[1],m=this.strides[2],g=[i,of(o,c,h,this.padding),of(l,f,d,this.padding),of(u,m,p,this.padding),this.filters];"channelsLast"!==this.dataFormat&&(a=e$.transpose(a,[0,2,3,4,1]));let x=oh.conv3dTranspose(a,this.kernel.read(),g,this.strides,this.padding);return"channelsLast"!==this.dataFormat&&(x=e$.transpose(x,[0,4,1,2,3])),null!==this.bias&&(x=aG(x,this.bias.read(),this.dataFormat)),null!==this.activation&&(x=this.activation.apply(x)),x})}computeOutputShape(e){let t,r,n,a,s=(e=ss(e)).slice();"channelsFirst"===this.dataFormat?(t=1,r=2,n=3,a=4):(t=4,r=1,n=2,a=3);let i=this.kernelSize[0],o=this.kernelSize[1],l=this.kernelSize[2],u=this.strides[0],h=this.strides[1],d=this.strides[2];return s[t]=this.filters,s[r]=of(s[r],u,i,this.padding),s[n]=of(s[n],h,o,this.padding),s[a]=of(s[a],d,l,this.padding),s}getConfig(){let e=super.getConfig();return delete e.dilationRate,e}}oI.className="Conv3DTranspose",nJ.registerClass(oI);class ok extends ob{constructor(e,t){if(super(e,t),this.DEFAULT_DEPTHWISE_INITIALIZER="glorotUniform",this.DEFAULT_POINTWISE_INITIALIZER="glorotUniform",this.depthwiseKernel=null,this.pointwiseKernel=null,null==t.filters)throw new nF("The `filters` configuration field is required by SeparableConv, but is unspecified.");if(null!=t.kernelInitializer||null!=t.kernelRegularizer||null!=t.kernelConstraint)throw new nF("Fields kernelInitializer, kernelRegularizer and kernelConstraint are invalid for SeparableConv2D. Use depthwiseInitializer, depthwiseRegularizer, depthwiseConstraint, pointwiseInitializer, pointwiseRegularizer and pointwiseConstraint instead.");if(null!=t.padding&&"same"!==t.padding&&"valid"!==t.padding)throw new nF(`SeparableConv${this.rank}D supports only padding modes: 'same' and 'valid', but received ${JSON.stringify(t.padding)}`);this.depthMultiplier=null==t.depthMultiplier?1:t.depthMultiplier,this.depthwiseInitializer=st(t.depthwiseInitializer||this.DEFAULT_DEPTHWISE_INITIALIZER),this.depthwiseRegularizer=on(t.depthwiseRegularizer),this.depthwiseConstraint=sD(t.depthwiseConstraint),this.pointwiseInitializer=st(t.depthwiseInitializer||this.DEFAULT_POINTWISE_INITIALIZER),this.pointwiseRegularizer=on(t.pointwiseRegularizer),this.pointwiseConstraint=sD(t.pointwiseConstraint)}build(e){if((e=ss(e)).length<this.rank+2)throw new nF(`Inputs to SeparableConv${this.rank}D should have rank ${this.rank+2}, but received input shape: ${JSON.stringify(e)}`);let t="channelsFirst"===this.dataFormat?1:e.length-1;if(null==e[t]||e[t]<0)throw new nF(`The channel dimension of the inputs should be defined, but found ${JSON.stringify(e[t])}`);let r=e[t],n=this.kernelSize.concat([r,this.depthMultiplier]),a=[];for(let e=0;e<this.rank;++e)a.push(1);a.push(r*this.depthMultiplier,this.filters),this.depthwiseKernel=this.addWeight("depthwise_kernel",n,"float32",this.depthwiseInitializer,this.depthwiseRegularizer,!0,this.depthwiseConstraint),this.pointwiseKernel=this.addWeight("pointwise_kernel",a,"float32",this.pointwiseInitializer,this.pointwiseRegularizer,!0,this.pointwiseConstraint),this.useBias?this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.inputSpec=[new sp({ndim:this.rank+2,axes:{[t]:r}})],this.built=!0}call(e,t){return(0,n$.tidy)(()=>{let t;if(e=sa(e),1===this.rank)throw new nD("1D separable convolution is not implemented yet.");return 2===this.rank&&("channelsFirst"===this.dataFormat&&(e=e$.transpose(e,[0,2,3,1])),t=ng.separableConv2d(e,this.depthwiseKernel.read(),this.pointwiseKernel.read(),this.strides,this.padding,this.dilationRate,"NHWC")),this.useBias&&(t=aG(t,this.bias.read(),this.dataFormat)),null!=this.activation&&(t=this.activation.apply(t)),"channelsFirst"===this.dataFormat&&(t=e$.transpose(t,[0,3,1,2])),t})}getConfig(){let e=super.getConfig();return delete e.rank,delete e.kernelInitializer,delete e.kernelRegularizer,delete e.kernelConstraint,e.depthwiseInitializer=nV(this.depthwiseInitializer),e.pointwiseInitializer=nV(this.pointwiseInitializer),e.depthwiseRegularizer=nV(this.depthwiseRegularizer),e.pointwiseRegularizer=nV(this.pointwiseRegularizer),e.depthwiseConstraint=nV(this.depthwiseConstraint),e.pointwiseConstraint=nV(this.pointwiseConstraint),e}}ok.className="SeparableConv";class oS extends ok{constructor(e){super(2,e)}}oS.className="SeparableConv2D",nJ.registerClass(oS);class oT extends ob{constructor(e){super(1,e),oT.verifyArgs(e),this.inputSpec=[{ndim:3}]}getConfig(){let e=super.getConfig();return delete e.rank,delete e.dataFormat,e}static verifyArgs(e){if("number"!=typeof e.kernelSize&&!nK(e.kernelSize,"number",1,1))throw new nF(`Conv1D expects config.kernelSize to be number or number[] with length 1, but received ${JSON.stringify(e.kernelSize)}.`)}}oT.className="Conv1D",nJ.registerClass(oT);class oN extends sx{constructor(e){super(e),"number"==typeof e.cropping?this.cropping=[[e.cropping,e.cropping],[e.cropping,e.cropping]]:"number"==typeof e.cropping[0]?this.cropping=[[e.cropping[0],e.cropping[0]],[e.cropping[1],e.cropping[1]]]:this.cropping=e.cropping,this.dataFormat=void 0===e.dataFormat?"channelsLast":e.dataFormat,this.inputSpec=[{ndim:4}]}computeOutputShape(e){return"channelsFirst"===this.dataFormat?[e[0],e[1],e[2]-this.cropping[0][0]-this.cropping[0][1],e[3]-this.cropping[1][0]-this.cropping[1][1]]:[e[0],e[1]-this.cropping[0][0]-this.cropping[0][1],e[2]-this.cropping[1][0]-this.cropping[1][1],e[3]]}call(e,t){return(0,n$.tidy)(()=>{if(e=sa(e),"channelsLast"===this.dataFormat){let t=aD(e,this.cropping[0][0],e.shape[1]-this.cropping[0][0]-this.cropping[0][1],2);return aD(t,this.cropping[1][0],e.shape[2]-this.cropping[1][1]-this.cropping[1][0],3)}{let t=aD(e,this.cropping[0][0],e.shape[2]-this.cropping[0][0]-this.cropping[0][1],3);return aD(t,this.cropping[1][0],e.shape[3]-this.cropping[1][1]-this.cropping[1][0],4)}})}getConfig(){let e={cropping:this.cropping,dataFormat:this.dataFormat};return Object.assign(e,super.getConfig()),e}}oN.className="Cropping2D",nJ.registerClass(oN);class o$ extends sx{constructor(e){super(e),this.DEFAULT_SIZE=[2,2],this.inputSpec=[{ndim:4}],this.size=null==e.size?this.DEFAULT_SIZE:e.size,this.dataFormat=null==e.dataFormat?"channelsLast":e.dataFormat,n7(this.dataFormat),this.interpolation=null==e.interpolation?"nearest":e.interpolation,function(e){nX(n3,"InterpolationFormat",e)}(this.interpolation)}computeOutputShape(e){if("channelsFirst"===this.dataFormat){let t=null==e[2]?null:this.size[0]*e[2],r=null==e[3]?null:this.size[1]*e[3];return[e[0],e[1],t,r]}{let t=null==e[1]?null:this.size[0]*e[1],r=null==e[2]?null:this.size[1]*e[2];return[e[0],t,r,e[3]]}}call(e,t){return n$.tidy(()=>{let t=sa(e),r=t.shape;if("channelsFirst"===this.dataFormat){t=e$.transpose(t,[0,2,3,1]);let e=this.size[0]*r[2],n=this.size[1]*r[3],a="nearest"===this.interpolation?al.image.resizeNearestNeighbor(t,[e,n]):al.image.resizeBilinear(t,[e,n]);return e$.transpose(a,[0,3,1,2])}{let e=this.size[0]*r[1],n=this.size[1]*r[2];return"nearest"===this.interpolation?al.image.resizeNearestNeighbor(t,[e,n]):al.image.resizeBilinear(t,[e,n])}})}getConfig(){let e={size:this.size,dataFormat:this.dataFormat,interpolation:this.interpolation};return Object.assign(e,super.getConfig()),e}}o$.className="UpSampling2D",nJ.registerClass(o$);var nJ=nJ;class oR extends oy{constructor(e){super(2,e),this.depthwiseKernel=null,this.depthMultiplier=null==e.depthMultiplier?1:e.depthMultiplier,this.depthwiseInitializer=st(e.depthwiseInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.depthwiseConstraint=sD(e.depthwiseConstraint),this.depthwiseRegularizer=on(e.depthwiseRegularizer)}build(e){if((e=ss(e)).length<4)throw new nF(`Inputs to DepthwiseConv2D should have rank 4. Received input shape: ${JSON.stringify(e)}.`);let t="channelsFirst"===this.dataFormat?1:3;if(null==e[t]||e[t]<0)throw new nF(`The channel dimension of the inputs to DepthwiseConv2D should be defined, but is not (${e[t]}).`);let r=e[t],n=[this.kernelSize[0],this.kernelSize[1],r,this.depthMultiplier];this.depthwiseKernel=this.addWeight("depthwise_kernel",n,null,this.depthwiseInitializer,this.depthwiseRegularizer,!0,this.depthwiseConstraint),this.useBias?this.bias=this.addWeight("bias",[r*this.depthMultiplier],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(e,t){return(0,n$.tidy)(()=>{let t=function(e,t,r=[1,1],n="valid",a){return(0,n$.tidy)(()=>{null==a&&(a=a$()),n7(a);let s=om(e,a);if(4!==e.rank)throw new nF(`Input for depthwiseConv2d is required to be 4-D, but is instead ${e.rank}-D`);if(4!==t.rank)throw new nF(`depthwiseKernel is required to be 4-D, but is instead ${t.rank}-D`);return s=rL.depthwiseConv2d(s,t,r,"same"===n?"same":"valid","NHWC",null),"channelsFirst"===a&&(s=e$.transpose(s,[0,3,1,2])),s})}(e=sa(e),this.depthwiseKernel.read(),this.strides,this.padding,this.dataFormat);return this.useBias&&(t=aG(t,this.bias.read(),this.dataFormat)),null!=this.activation&&(t=this.activation.apply(t)),t})}computeOutputShape(e){e=ss(e);let t="channelsFirst"===this.dataFormat?e[2]:e[1],r="channelsFirst"===this.dataFormat?e[3]:e[2],n="channelsFirst"===this.dataFormat?e[1]*this.depthMultiplier:e[3]*this.depthMultiplier,a=oc(t,this.kernelSize[0],this.padding,this.strides[0]),s=oc(r,this.kernelSize[1],this.padding,this.strides[1]);return"channelsFirst"===this.dataFormat?[e[0],n,a,s]:[e[0],a,s,n]}getConfig(){let e=super.getConfig();return e.depthMultiplier=this.depthMultiplier,e.depthwiseInitializer=nV(this.depthwiseInitializer),e.depthwiseRegularizer=nV(this.depthwiseRegularizer),e.depthwiseConstraint=nV(this.depthwiseRegularizer),e}}oR.className="DepthwiseConv2D",nJ.registerClass(oR);var nJ=nJ,nJ=nJ;function oA(e,t,r,n){if(Array.isArray(e)){if(null!=t||null!=r)throw new nF("When inputs is an array, neither initialState or constants should be provided");null!=n&&(r=e.slice(e.length-n,e.length),e=e.slice(0,e.length-n)),e.length>1&&(t=e.slice(1,e.length)),e=e[0]}function a(e){return null==e||Array.isArray(e)?e:[e]}return{inputs:e,initialState:t=a(t),constants:r=a(r)}}function oE(e,t,r,n=!1,a,s,i=!1,o=!1){return n$.tidy(()=>{let l,u,h,d=t.shape.length;if(d<3)throw new nF(`Input should be at least 3D, but is ${d}D.`);let p=[1,0].concat(aT(2,d));if(t=e$.transpose(t,p),null!=s)throw new nD("The rnn() functoin of the deeplearn.js backend does not support constants yet.");i&&console.warn("Backend rnn(): the unroll = true option is not applicable to the imperative deeplearn.js backend."),null!=a&&((a=y.cast(y.cast(a,"bool"),"float32")).rank===d-1&&(a=rl.expandDims(a,-1)),a=e$.transpose(a,p)),n&&(t=t_.reverse(t,0),null!=a&&(a=t_.reverse(a,0)));let c=[],f=r,m=t.shape[0],g=tv.unstack(t);null!=a&&(u=tv.unstack(a));for(let t=0;t<m;++t){let r=g[t],n=n$.tidy(()=>e(r,f));if(null==a)l=n[0],f=n[1];else{let e=n$.tidy(()=>{let e=u[t],r=N.sub(na.onesLike(e),e);return{output:P.add(b.mul(n[0],e),b.mul(f[0],r)),newStates:f.map((t,a)=>P.add(b.mul(n[1][a],e),b.mul(t,r)))}});l=e.output,f=e.newStates}o&&c.push(l)}return o&&(h=eH.stack(c,1)),[l,h,f]})}class oF extends sx{constructor(e){let t;if(super(e),null==e.cell)throw new nF("cell property is missing for the constructor of RNN.");if(null==(t=Array.isArray(e.cell)?new oB({cells:e.cell}):e.cell).stateSize)throw new nF("The RNN cell should have an attribute `stateSize` (tuple of integers, one integer per RNN state).");this.cell=t,this.returnSequences=null!=e.returnSequences&&e.returnSequences,this.returnState=null!=e.returnState&&e.returnState,this.goBackwards=null!=e.goBackwards&&e.goBackwards,this._stateful=null!=e.stateful&&e.stateful,this.unroll=null!=e.unroll&&e.unroll,this.supportsMasking=!0,this.inputSpec=[new sp({ndim:3})],this.stateSpec=null,this.states_=null,this.numConstants=null,this.keptStates=[]}getStates(){return null==this.states_?aT(0,Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1).map(e=>null):this.states_}setStates(e){this.states_=e}computeOutputShape(e){let t;sr(e)&&(e=e[0]);let r=this.cell.stateSize;Array.isArray(r)||(r=[r]);let n=r[0];if(t=this.returnSequences?[e[0],e[1],n]:[e[0],n],!this.returnState)return t;{let n=[];for(let t of r)n.push([e[0],t]);return[t].concat(n)}}computeMask(e,t){return n$.tidy(()=>{Array.isArray(t)&&(t=t[0]);let e=this.returnSequences?t:null;return this.returnState?[e].concat(this.states.map(e=>null)):e})}get states(){if(null!=this.states_)return this.states_;{let e=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1,t=[];for(let r=0;r<e;++r)t.push(null);return t}}set states(e){this.states_=e}build(e){let t;if(null!=this.numConstants)throw new nD("Constants support is not implemented in RNN yet.");sr(e)&&(e=e[0]);let r=this.stateful?e[0]:null,n=e.slice(2);this.inputSpec[0]=new sp({shape:[r,null,...n]});let a=[e[0]].concat(e.slice(2));if(this.cell.build(a),t=Array.isArray(this.cell.stateSize)?this.cell.stateSize:[this.cell.stateSize],null!=this.stateSpec){if(!nR.util.arraysEqual(this.stateSpec.map(e=>e.shape[e.shape.length-1]),t))throw new nF(`An initialState was passed that is not compatible with cell.stateSize. Received stateSpec=${this.stateSpec}; However cell.stateSize is ${this.cell.stateSize}`)}else this.stateSpec=t.map(e=>new sp({shape:[null,e]}));this.stateful&&this.resetStates()}resetStates(e,t=!1){(0,n$.tidy)(()=>{if(!this.stateful)throw new nA("Cannot call resetStates() on an RNN Layer that is not stateful.");let r=this.inputSpec[0].shape[0];if(null==r)throw new nF("If an RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batchInputShape` option to your first layer.\n- If using the functional API, specify the batch size by passing a `batchShape` option to your Input layer.");if(null==this.states_)Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(e=>tx.zeros([r,e])):this.states_=[tx.zeros([r,this.cell.stateSize])];else if(null==e)n$.dispose(this.states_),null!=this.keptStates&&(n$.dispose(this.keptStates),this.keptStates=[]),Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(e=>tx.zeros([r,e])):this.states_[0]=tx.zeros([r,this.cell.stateSize]);else{if(Array.isArray(e)||(e=[e]),e.length!==this.states_.length)throw new nF(`Layer ${this.name} expects ${this.states_.length} state(s), but it received ${e.length} state value(s). Input received: ${e}`);!0===t?this.keptStates.push(this.states_.slice()):n$.dispose(this.states_);for(let t=0;t<this.states_.length;++t){let n=e[t],a=[r,Array.isArray(this.cell.stateSize)?this.cell.stateSize[t]:this.cell.stateSize];if(!nR.util.arraysEqual(n.shape,a))throw new nF(`State ${t} is incompatible with layer ${this.name}: expected shape=${a}, received shape=${n.shape}`);this.states_[t]=n}}this.states_=this.states_.map(e=>n$.keep(e.clone()))})}apply(e,t){let r=null==t?null:t.initialState,n=null==t?null:t.constants;null==t&&(t={});let a=oA(e,r,n,this.numConstants);e=a.inputs,r=a.initialState,n=a.constants;let s=[],i=[];if(null!=r){for(let e of(t.initialState=r,s=s.concat(r),this.stateSpec=[],r))this.stateSpec.push(new sp({shape:e.shape}));i=i.concat(this.stateSpec)}if(null!=n&&(t.constants=n,s=s.concat(n),this.numConstants=n.length),!(s[0]instanceof sc))return super.apply(e,t);{let r=[e].concat(s),n=this.inputSpec.concat(i),a=this.inputSpec;this.inputSpec=n;let o=super.apply(r,t);return this.inputSpec=a,o}}call(e,t){return(0,n$.tidy)(()=>{let r=null==t?null:t.mask,n=null==t?null:t.training,a=null==t?null:t.initialState;e=sa(e),null==a&&(a=this.stateful?this.states_:this.getInitialState(e));let s=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1;if(a.length!==s)throw new nF(`RNN Layer has ${s} state(s) but was passed ${a.length} initial state(s).`);this.unroll&&console.warn("Ignoring unroll = true for RNN layer, due to imperative backend.");let i={training:n},o=oE((e,t)=>{let r=this.cell.call([e].concat(t),i);return[r[0],r.slice(1)]},e,a,this.goBackwards,r,null,this.unroll,this.returnSequences),l=o[0],u=o[1],h=o[2];this.stateful&&this.resetStates(h,n);let d=this.returnSequences?u:l;return this.returnState?[d].concat(h):d})}getInitialState(e){return(0,n$.tidy)(()=>{let t=tx.zeros(e.shape);return(t=aA(t=F.sum(t,[1,2])),Array.isArray(this.cell.stateSize))?this.cell.stateSize.map(e=>e>1?az(t,[1,e]):t):this.cell.stateSize>1?[az(t,[1,this.cell.stateSize])]:[t]})}get trainableWeights(){return this.trainable?this.cell.trainableWeights:[]}get nonTrainableWeights(){return this.trainable?this.cell.nonTrainableWeights:this.cell.weights}setFastWeightInitDuringBuild(e){super.setFastWeightInitDuringBuild(e),null!=this.cell&&this.cell.setFastWeightInitDuringBuild(e)}getConfig(){let e=super.getConfig(),t={returnSequences:this.returnSequences,returnState:this.returnState,goBackwards:this.goBackwards,stateful:this.stateful,unroll:this.unroll};null!=this.numConstants&&(t.numConstants=this.numConstants);let r=this.cell.getConfig();return this.getClassName()===oF.className&&(t.cell={className:this.cell.getClassName(),config:r}),Object.assign(Object.assign(Object.assign({},r),e),t)}static fromConfig(e,t,r={}){let n=sK(t.cell,r);return new e(Object.assign(t,{cell:n}))}}oF.className="RNN",nJ.registerClass(oF);class oD extends sx{}class oO extends oD{constructor(e){super(e),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",this.units=e.units,nY(this.units,"units"),this.activation=i8(null==e.activation?this.DEFAULT_ACTIVATION:e.activation),this.useBias=null==e.useBias||e.useBias,this.kernelInitializer=st(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=st(e.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=st(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelRegularizer=on(e.kernelRegularizer),this.recurrentRegularizer=on(e.recurrentRegularizer),this.biasRegularizer=on(e.biasRegularizer),this.kernelConstraint=sD(e.kernelConstraint),this.recurrentConstraint=sD(e.recurrentConstraint),this.biasConstraint=sD(e.biasConstraint),this.dropout=ak([1,aS([0,null==e.dropout?0:e.dropout])]),this.recurrentDropout=ak([1,aS([0,null==e.recurrentDropout?0:e.recurrentDropout])]),this.dropoutFunc=e.dropoutFunc,this.stateSize=this.units,this.dropoutMask=null,this.recurrentDropoutMask=null}build(e){e=ss(e),this.kernel=this.addWeight("kernel",[e[e.length-1],this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,this.units],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias?this.bias=this.addWeight("bias",[this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(e,t){return(0,n$.tidy)(()=>{let r;if(2!==e.length)throw new nF(`SimpleRNNCell expects 2 input Tensors, got ${e.length}.`);let n=e[1];e=e[0];let a=null!=t.training&&t.training;0<this.dropout&&this.dropout<1&&null==this.dropoutMask&&(this.dropoutMask=oW({ones:()=>na.onesLike(e),rate:this.dropout,training:a,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&null==this.recurrentDropoutMask&&(this.recurrentDropoutMask=oW({ones:()=>na.onesLike(n),rate:this.recurrentDropout,training:a,dropoutFunc:this.dropoutFunc}));let s=this.dropoutMask,i=this.recurrentDropoutMask;r=null!=s?aM(b.mul(e,s),this.kernel.read()):aM(e,this.kernel.read()),null!=this.bias&&(r=aG(r,this.bias.read())),null!=i&&(n=b.mul(n,i));let o=P.add(r,aM(n,this.recurrentKernel.read()));return null!=this.activation&&(o=this.activation.apply(o)),[o,o]})}getConfig(){let e=super.getConfig(),t={units:this.units,activation:i5(this.activation),useBias:this.useBias,kernelInitializer:nV(this.kernelInitializer),recurrentInitializer:nV(this.recurrentInitializer),biasInitializer:nV(this.biasInitializer),kernelRegularizer:nV(this.kernelRegularizer),recurrentRegularizer:nV(this.recurrentRegularizer),biasRegularizer:nV(this.biasRegularizer),activityRegularizer:nV(this.activityRegularizer),kernelConstraint:nV(this.kernelConstraint),recurrentConstraint:nV(this.recurrentConstraint),biasConstraint:nV(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout};return Object.assign(Object.assign({},e),t)}}oO.className="SimpleRNNCell",nJ.registerClass(oO);class oL extends oF{constructor(e){e.cell=new oO(e),super(e)}call(e,t){return(0,n$.tidy)(()=>{null!=this.cell.dropoutMask&&(n$.dispose(this.cell.dropoutMask),this.cell.dropoutMask=null),null!=this.cell.recurrentDropoutMask&&(n$.dispose(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);let r=null==t?null:t.mask,n=null==t?null:t.training,a=null==t?null:t.initialState;return super.call(e,{mask:r,training:n,initialState:a})})}static fromConfig(e,t){return new e(t)}}oL.className="SimpleRNN",nJ.registerClass(oL);class oz extends oD{constructor(e){if(super(e),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_RECURRENT_ACTIVATION="hardSigmoid",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",e.resetAfter)throw new nF("GRUCell does not support reset_after parameter set to true.");this.units=e.units,nY(this.units,"units"),this.activation=i8(void 0===e.activation?this.DEFAULT_ACTIVATION:e.activation),this.recurrentActivation=i8(void 0===e.recurrentActivation?this.DEFAULT_RECURRENT_ACTIVATION:e.recurrentActivation),this.useBias=null==e.useBias||e.useBias,this.kernelInitializer=st(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=st(e.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=st(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelRegularizer=on(e.kernelRegularizer),this.recurrentRegularizer=on(e.recurrentRegularizer),this.biasRegularizer=on(e.biasRegularizer),this.kernelConstraint=sD(e.kernelConstraint),this.recurrentConstraint=sD(e.recurrentConstraint),this.biasConstraint=sD(e.biasConstraint),this.dropout=ak([1,aS([0,null==e.dropout?0:e.dropout])]),this.recurrentDropout=ak([1,aS([0,null==e.recurrentDropout?0:e.recurrentDropout])]),this.dropoutFunc=e.dropoutFunc,this.implementation=e.implementation,this.stateSize=this.units,this.dropoutMask=null,this.recurrentDropoutMask=null}build(e){let t=(e=ss(e))[e.length-1];this.kernel=this.addWeight("kernel",[t,3*this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,3*this.units],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias?this.bias=this.addWeight("bias",[3*this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(e,t){return(0,n$.tidy)(()=>{let r,n,a;if(2!==e.length)throw new nF(`GRUCell expects 2 input Tensors (inputs, h, c), got ${e.length}.`);let s=null!=t.training&&t.training,i=e[1];e=e[0],0<this.dropout&&this.dropout<1&&null==this.dropoutMask&&(this.dropoutMask=oW({ones:()=>na.onesLike(e),rate:this.dropout,training:s,count:3,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&null==this.recurrentDropoutMask&&(this.recurrentDropoutMask=oW({ones:()=>na.onesLike(i),rate:this.recurrentDropout,training:s,count:3,dropoutFunc:this.dropoutFunc}));let o=this.dropoutMask,l=this.recurrentDropoutMask;0<this.dropout&&this.dropout<1&&(e=b.mul(e,o[0]));let u=aM(e,this.kernel.read());this.useBias&&(u=aG(u,this.bias.read())),0<this.recurrentDropout&&this.recurrentDropout<1&&(i=b.mul(i,l[0]));let h=this.recurrentKernel.read(),[d,p]=ep.split(h,[2*this.units,this.units],h.rank-1),c=aM(i,d),[f,m,g]=ep.split(u,3,u.rank-1),[x,y]=ep.split(c,2,c.rank-1);r=this.recurrentActivation.apply(P.add(f,x)),n=this.recurrentActivation.apply(P.add(m,y));let v=aM(b.mul(n,i),p);a=this.activation.apply(P.add(g,v));let w=P.add(b.mul(r,i),b.mul(P.add(1,I.neg(r)),a));return[w,w]})}getConfig(){let e=super.getConfig(),t={units:this.units,activation:i5(this.activation),recurrentActivation:i5(this.recurrentActivation),useBias:this.useBias,kernelInitializer:nV(this.kernelInitializer),recurrentInitializer:nV(this.recurrentInitializer),biasInitializer:nV(this.biasInitializer),kernelRegularizer:nV(this.kernelRegularizer),recurrentRegularizer:nV(this.recurrentRegularizer),biasRegularizer:nV(this.biasRegularizer),activityRegularizer:nV(this.activityRegularizer),kernelConstraint:nV(this.kernelConstraint),recurrentConstraint:nV(this.recurrentConstraint),biasConstraint:nV(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout,implementation:this.implementation,resetAfter:!1};return Object.assign(Object.assign({},e),t)}}oz.className="GRUCell",nJ.registerClass(oz);class o_ extends oF{constructor(e){0===e.implementation&&console.warn("`implementation=0` has been deprecated, and now defaults to `implementation=1`. Please update your layer call."),e.cell=new oz(e),super(e)}call(e,t){return(0,n$.tidy)(()=>{null!=this.cell.dropoutMask&&(n$.dispose(this.cell.dropoutMask),this.cell.dropoutMask=null),null!=this.cell.recurrentDropoutMask&&(n$.dispose(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);let r=null==t?null:t.mask,n=null==t?null:t.training,a=null==t?null:t.initialState;return super.call(e,{mask:r,training:n,initialState:a})})}static fromConfig(e,t){return 0===t.implmentation&&(t.implementation=1),new e(t)}}o_.className="GRU",nJ.registerClass(o_);class oM extends oD{constructor(e){super(e),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_RECURRENT_ACTIVATION="hardSigmoid",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",this.units=e.units,nY(this.units,"units"),this.activation=i8(void 0===e.activation?this.DEFAULT_ACTIVATION:e.activation),this.recurrentActivation=i8(void 0===e.recurrentActivation?this.DEFAULT_RECURRENT_ACTIVATION:e.recurrentActivation),this.useBias=null==e.useBias||e.useBias,this.kernelInitializer=st(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=st(e.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=st(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.unitForgetBias=e.unitForgetBias,this.kernelRegularizer=on(e.kernelRegularizer),this.recurrentRegularizer=on(e.recurrentRegularizer),this.biasRegularizer=on(e.biasRegularizer),this.kernelConstraint=sD(e.kernelConstraint),this.recurrentConstraint=sD(e.recurrentConstraint),this.biasConstraint=sD(e.biasConstraint),this.dropout=ak([1,aS([0,null==e.dropout?0:e.dropout])]),this.recurrentDropout=ak([1,aS([0,null==e.recurrentDropout?0:e.recurrentDropout])]),this.dropoutFunc=e.dropoutFunc,this.implementation=e.implementation,this.stateSize=[this.units,this.units],this.dropoutMask=null,this.recurrentDropoutMask=null}build(e){var t;let r,n=(e=ss(e))[e.length-1];if(this.kernel=this.addWeight("kernel",[n,4*this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,4*this.units],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias){if(this.unitForgetBias){let e=this.biasInitializer,n=this.units;r=new((t=class extends aj{apply(t,r){let a=e.apply([n]),s=new aK().apply([n]),i=e.apply([2*n]);return aL(aL(a,s),i)}}).className="CustomInit",t)}else r=this.biasInitializer;this.bias=this.addWeight("bias",[4*this.units],null,r,this.biasRegularizer,!0,this.biasConstraint)}else this.bias=null;this.built=!0}call(e,t){return(0,n$.tidy)(()=>{let r,n,a,s,i=null!=t.training&&t.training;if(3!==e.length)throw new nF(`LSTMCell expects 3 input Tensors (inputs, h, c), got ${e.length}.`);let o=e[1],l=e[2];e=e[0],0<this.dropout&&this.dropout<1&&null==this.dropoutMask&&(this.dropoutMask=oW({ones:()=>na.onesLike(e),rate:this.dropout,training:i,count:4,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&null==this.recurrentDropoutMask&&(this.recurrentDropoutMask=oW({ones:()=>na.onesLike(o),rate:this.recurrentDropout,training:i,count:4,dropoutFunc:this.dropoutFunc}));let u=this.dropoutMask,h=this.recurrentDropoutMask;0<this.dropout&&this.dropout<1&&(e=b.mul(e,u[0]));let d=aM(e,this.kernel.read());0<this.recurrentDropout&&this.recurrentDropout<1&&(o=b.mul(o,h[0])),d=P.add(d,aM(o,this.recurrentKernel.read())),this.useBias&&(d=aG(d,this.bias.read()));let[p,c,f,m]=ep.split(d,4,d.rank-1);r=this.recurrentActivation.apply(p),n=this.recurrentActivation.apply(c),a=P.add(b.mul(n,l),b.mul(r,this.activation.apply(f))),s=this.recurrentActivation.apply(m);let g=b.mul(s,this.activation.apply(a));return[g,g,a]})}getConfig(){let e=super.getConfig(),t={units:this.units,activation:i5(this.activation),recurrentActivation:i5(this.recurrentActivation),useBias:this.useBias,kernelInitializer:nV(this.kernelInitializer),recurrentInitializer:nV(this.recurrentInitializer),biasInitializer:nV(this.biasInitializer),unitForgetBias:this.unitForgetBias,kernelRegularizer:nV(this.kernelRegularizer),recurrentRegularizer:nV(this.recurrentRegularizer),biasRegularizer:nV(this.biasRegularizer),activityRegularizer:nV(this.activityRegularizer),kernelConstraint:nV(this.kernelConstraint),recurrentConstraint:nV(this.recurrentConstraint),biasConstraint:nV(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout,implementation:this.implementation};return Object.assign(Object.assign({},e),t)}}oM.className="LSTMCell",nJ.registerClass(oM);class oP extends oF{constructor(e){0===e.implementation&&console.warn("`implementation=0` has been deprecated, and now defaults to `implementation=1`. Please update your layer call."),e.cell=new oM(e),super(e)}call(e,t){return(0,n$.tidy)(()=>{null!=this.cell.dropoutMask&&(n$.dispose(this.cell.dropoutMask),this.cell.dropoutMask=null),null!=this.cell.recurrentDropoutMask&&(n$.dispose(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);let r=null==t?null:t.mask,n=null==t?null:t.training,a=null==t?null:t.initialState;return super.call(e,{mask:r,training:n,initialState:a})})}static fromConfig(e,t){return 0===t.implmentation&&(t.implementation=1),new e(t)}}oP.className="LSTM",nJ.registerClass(oP);class oB extends oD{constructor(e){super(e),this.cells=e.cells}get stateSize(){let e=[];for(let t of this.cells.slice().reverse())Array.isArray(t.stateSize)?e.push(...t.stateSize):e.push(t.stateSize);return e}call(e,t){return(0,n$.tidy)(()=>{let r,n=e.slice(1),a=[];for(let e of this.cells.slice().reverse())Array.isArray(e.stateSize)?a.push(n.splice(0,e.stateSize.length)):a.push(n.splice(0,1));a.reverse();let s=[];for(let i=0;i<this.cells.length;++i){let o=this.cells[i];n=a[i],r=0===i?[e[0]].concat(n):[r[0]].concat(n),r=o.call(r,t),s.push(r.slice(1))}for(let e of(n=[],s.slice().reverse()))n.push(...e);return[r[0]].concat(n)})}build(e){let t;sr(e)&&(e=e[0]),this.cells.forEach((r,n)=>{ar(`RNNCell_${n}`,()=>{r.build(e),t=Array.isArray(r.stateSize)?r.stateSize[0]:r.stateSize,e=[e[0],t]})}),this.built=!0}getConfig(){let e=super.getConfig(),t=this.cells.map(e=>({className:e.getClassName(),config:e.getConfig()}));return Object.assign(Object.assign({},e),{cells:t})}static fromConfig(e,t,r={}){let n=[];for(let e of t.cells)n.push(sK(e,r));return new e({cells:n})}get trainableWeights(){if(!this.trainable)return[];let e=[];for(let t of this.cells)e.push(...t.trainableWeights);return e}get nonTrainableWeights(){let e=[];for(let t of this.cells)e.push(...t.nonTrainableWeights);if(!this.trainable){let t=[];for(let e of this.cells)t.push(...e.trainableWeights);return t.concat(e)}return e}getWeights(){let e=[];for(let t of this.cells)e.push(...t.weights);return sh(e)}setWeights(e){let t=[];for(let r of this.cells){let n=r.weights.length,a=e.splice(n);for(let e=0;e<r.weights.length;++e)t.push([r.weights[e],a[e]])}sd(t)}}function oW(e){let{ones:t,rate:r,training:n=!1,count:a=1,dropoutFunc:s}=e,i=()=>null!=s?s(t(),r):aU(t(),r),o=()=>aV(i,t,n);return!a||a<=1?n$.keep(o().clone()):Array(a).fill(void 0).map(o).map(e=>n$.keep(e.clone()))}oB.className="StackedRNNCells",nJ.registerClass(oB);var oG=function(e,t){var r={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&0>t.indexOf(n)&&(r[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var a=0,n=Object.getOwnPropertySymbols(e);a<n.length;a++)0>t.indexOf(n[a])&&Object.prototype.propertyIsEnumerable.call(e,n[a])&&(r[n[a]]=e[n[a]]);return r};class oU extends oF{constructor(e){if(e.unroll)throw new nD("Unrolling is not possible with convolutional RNNs.");if(Array.isArray(e.cell))throw new nD("It is not possible at the moment to stack convolutional cells.");super(e),this.inputSpec=[new sp({ndim:5})]}call(e,t){return n$.tidy(()=>{if(null!=this.cell.dropoutMask&&(n$.dispose(this.cell.dropoutMask),this.cell.dropoutMask=null),null!=this.cell.recurrentDropoutMask&&(n$.dispose(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null),t&&t.constants)throw new nF("ConvRNN2D cell does not support constants");let r=null==t?null:t.mask,n=null==t?null:t.training,a=null==t?null:t.initialState;return super.call(e,{mask:r,training:n,initialState:a})})}computeOutputShape(e){let t=this.computeSingleOutputShape(e);return this.returnSequences||(t=[t[0],...t.slice(2)]),this.returnState&&(t=[t,...[,,].fill([e[0],...t.slice(-3)])]),t}getInitialState(e){return n$.tidy(()=>{let{stateSize:t}=this.cell,r=e.shape,n=this.computeSingleOutputShape(r),a=[n[0],...n.slice(2)],s=tx.zeros(a);return Array.isArray(t)?Array(t.length).fill(s):[s]})}resetStates(e,t=!1){n$.tidy(()=>{if(!this.stateful)throw new nA("Cannot call resetStates() on an RNN Layer that is not stateful.");let r=this.inputSpec[0].shape,n=this.computeSingleOutputShape(r),a=[n[0],...n.slice(2)];if(null==r[0])throw new nF("If an RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batchInputShape` option to your first layer.\n- If using the functional API, specify the batch size by passing a `batchShape` option to your Input layer.");if(null==this.getStates())Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(()=>tx.zeros(a)):this.states_=[tx.zeros(a)];else if(null==e)n$.dispose(this.states_),null!=this.keptStates&&(n$.dispose(this.keptStates),this.keptStates=[]),Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(()=>tx.zeros(a)):this.states_[0]=tx.zeros(a);else{if(Array.isArray(e)||(e=[e]),e.length!==this.states_.length)throw new nF(`Layer ${this.name} expects ${this.states_.length} state(s), but it received ${e.length} state value(s). Input received: ${e}`);t?this.keptStates.push(this.states_.slice()):n$.dispose(this.states_);for(let t=0;t<this.states_.length;++t){let r=e[t];if(!nR.util.arraysEqual(r.shape,a))throw new nF(`State ${t} is incompatible with layer ${this.name}: expected shape=${a}, received shape=${r.shape}`);this.states_[t]=r}}this.states_=this.states_.map(e=>n$.keep(e.clone()))})}computeSingleOutputShape(e){let{dataFormat:t,filters:r,kernelSize:n,padding:a,strides:s,dilationRate:i}=this.cell,o="channelsFirst"===t,l=e[o?3:2],u=e[o?4:3],h=oc(l,n[0],a,s[0],i[0]),d=oc(u,n[1],a,s[1],i[1]);return[...e.slice(0,2),...o?[r,h,d]:[h,d,r]]}}oU.className="ConvRNN2D";class oV extends oM{constructor(e){const{filters:t,kernelSize:r,strides:n,padding:a,dataFormat:s,dilationRate:i}=e;super(Object.assign(Object.assign({},e),{units:t})),this.filters=t,nY(this.filters,"filters"),this.kernelSize=op(r,2,"kernelSize"),this.kernelSize.forEach(e=>nY(e,"kernelSize")),this.strides=op(n||1,2,"strides"),this.strides.forEach(e=>nY(e,"strides")),this.padding=a||"valid",n9(this.padding),this.dataFormat=s||"channelsLast",n7(this.dataFormat),this.dilationRate=op(i||1,2,"dilationRate"),this.dilationRate.forEach(e=>nY(e,"dilationRate"))}build(e){var t;e=ss(e);let r="channelsFirst"===this.dataFormat?1:e.length-1;if(null==e[r])throw new nF(`The channel dimension of the input should be defined. Found ${e[r]}`);let n=e[r],a=this.kernelSize.concat([n,4*this.filters]);this.kernel=this.addWeight("kernel",a,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint);let s=this.kernelSize.concat([this.filters,4*this.filters]);if(this.recurrentKernel=this.addWeight("recurrent_kernel",s,null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias){let e;if(this.unitForgetBias){let r=this.biasInitializer,n=this.filters;e=new((t=class extends aj{apply(e,t){return aO([r.apply([n]),to.ones([n]),r.apply([2*n])])}}).className="CustomInit",t)}else e=this.biasInitializer;this.bias=this.addWeight("bias",[4*this.filters],null,e,this.biasRegularizer,!0,this.biasConstraint)}this.built=!0}call(e,t){return n$.tidy(()=>{if(3!==e.length)throw new nF(`ConvLSTM2DCell expects 3 input Tensors (inputs, h, c), got ${e.length}.`);let r=t.training||!1,n=e[0],a=e[1],s=e[2];0<this.dropout&&this.dropout<1&&null==this.dropoutMask&&(this.dropoutMask=oW({ones:()=>na.onesLike(n),rate:this.dropout,training:r,count:4,dropoutFunc:this.dropoutFunc}));let i=this.dropoutMask,o=(e,t,r)=>t&&t[r]?b.mul(t[r],e):e,l=o(n,i,0),u=o(n,i,1),h=o(n,i,2),d=o(n,i,3);0<this.recurrentDropout&&this.recurrentDropout<1&&null==this.recurrentDropoutMask&&(this.recurrentDropoutMask=oW({ones:()=>na.onesLike(a),rate:this.recurrentDropout,training:r,count:4,dropoutFunc:this.dropoutFunc}));let p=this.recurrentDropoutMask,c=o(a,p,0),f=o(a,p,1),m=o(a,p,2),g=o(a,p,3),[x,y,v,w]=ep.split(this.kernel.read(),4,3),[C,I,k,S]=this.useBias?ep.split(this.bias.read(),4):[null,null,null,null];l=this.inputConv(l,x,C,this.padding),u=this.inputConv(u,y,I,this.padding),h=this.inputConv(h,v,k,this.padding),d=this.inputConv(d,w,S,this.padding);let[T,N,$,R]=ep.split(this.recurrentKernel.read(),4,3);c=this.recurrentConv(c,T),f=this.recurrentConv(f,N),m=this.recurrentConv(m,$),g=this.recurrentConv(g,R);let A=this.recurrentActivation.apply(P.add(l,c)),E=this.recurrentActivation.apply(P.add(u,f)),F=P.add(b.mul(E,s),b.mul(A,this.activation.apply(P.add(h,m)))),D=b.mul(this.recurrentActivation.apply(P.add(d,g)),this.activation.apply(F));return[D,D,F]})}getConfig(){let e=super.getConfig(),{units:t}=e,r=oG(e,["units"]),n={filters:this.filters,kernelSize:this.kernelSize,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,strides:this.strides};return Object.assign(Object.assign({},r),n)}inputConv(e,t,r,n){let a=ex.conv2d(e,t,this.strides,n||"valid","channelsFirst"===this.dataFormat?"NCHW":"NHWC",this.dilationRate);return r?aG(a,r,this.dataFormat):a}recurrentConv(e,t){return ex.conv2d(e,t,1,"same","channelsFirst"===this.dataFormat?"NCHW":"NHWC")}}oV.className="ConvLSTM2DCell",nJ.registerClass(oV);class oH extends oU{constructor(e){const t=new oV(e);super(Object.assign(Object.assign({},e),{cell:t}))}static fromConfig(e,t){return new e(t)}}oH.className="ConvLSTM2D",nJ.registerClass(oH);var nJ=nJ;class oq extends sx{constructor(e){super(e),this.rate=Math.max(Math.min(e.rate,1),0),this.noiseShape=e.noiseShape,this.seed=e.seed,this.supportsMasking=!0}getNoiseShape(e){if(null==this.noiseShape)return this.noiseShape;let t=e.shape,r=[];for(let e=0;e<this.noiseShape.length;++e)r.push(null==this.noiseShape[e]?t[e]:this.noiseShape[e]);return r}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t);let r=sa(e);if(0<this.rate&&this.rate<1){let e=null!=t.training&&t.training,n=this.getNoiseShape(r);return aV(()=>aU(r,this.rate,n,this.seed),()=>r,e)}return e})}getConfig(){let e={rate:this.rate,noiseShape:this.noiseShape,seed:this.seed};return Object.assign(e,super.getConfig()),e}dispose(){return super.dispose()}}oq.className="Dropout",nJ.registerClass(oq);class oj extends oq{constructor(e){super(e),this.inputSpec=[{ndim:3}]}getNoiseShape(e){let t=e.shape;return[t[0],1,t[2]]}}oj.className="SpatialDropout1D",nJ.registerClass(oj);class oX extends sx{constructor(e){if(super(e),this.activation=null,this.useBias=!0,this.kernel=null,this.bias=null,this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_BIAS_INITIALIZER="zeros",null==e.batchInputShape&&null==e.inputShape&&null!=e.inputDim){let t=null;null!=e.batchSize&&(t=e.batchSize),this.batchInputShape=[t,e.inputDim]}this.units=e.units,nY(this.units,"units"),this.activation=i8(e.activation),null!=e.useBias&&(this.useBias=e.useBias),this.kernelInitializer=st(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.biasInitializer=st(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelConstraint=sD(e.kernelConstraint),this.biasConstraint=sD(e.biasConstraint),this.kernelRegularizer=on(e.kernelRegularizer),this.biasRegularizer=on(e.biasRegularizer),this.activityRegularizer=on(e.activityRegularizer),this.supportsMasking=!0,this.inputSpec=[{minNDim:2}]}build(e){let t=(e=ss(e))[e.length-1];null==this.kernel&&(this.kernel=this.addWeight("kernel",[t,this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint))),this.inputSpec=[{minNDim:2,axes:{[-1]:t}}],this.built=!0}computeOutputShape(e){let t=(e=ss(e)).slice();return t[t.length-1]=this.units,t}call(e,t){return(0,n$.tidy)(()=>{let r;this.invokeCallHook(e,t);let n=sa(e),a=nZ(this.activation.getClassName());return null!=a?r=aM(n,this.kernel.read(),a,this.bias?this.bias.read():null):(r=aM(n,this.kernel.read()),null!=this.bias&&(r=aG(r,this.bias.read())),null!=this.activation&&(r=this.activation.apply(r))),r})}getConfig(){let e={units:this.units,activation:i5(this.activation),useBias:this.useBias,kernelInitializer:nV(this.kernelInitializer),biasInitializer:nV(this.biasInitializer),kernelRegularizer:nV(this.kernelRegularizer),biasRegularizer:nV(this.biasRegularizer),activityRegularizer:nV(this.activityRegularizer),kernelConstraint:nV(this.kernelConstraint),biasConstraint:nV(this.biasConstraint)};return Object.assign(e,super.getConfig()),e}}oX.className="Dense",nJ.registerClass(oX);class oK extends sx{constructor(e){super(e=e||{}),this.inputSpec=[{minNDim:3}],this.dataFormat=e.dataFormat}computeOutputShape(e){for(let t of(e=ss(e)).slice(1))if(null==t)throw new nF(`The shape of the input to "Flatten" is not fully defined (got ${e.slice(1)}). Make sure to pass a complete "input_shape" or "batch_input_shape" argument to the first layer in your model.`);return[e[0],aI(e,1)]}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t);let r=sa(e);if("channelsFirst"===this.dataFormat&&r.rank>1){let e=[0];for(let t=2;t<r.rank;++t)e.push(t);e.push(1),r=(0,e$.transpose)(r,e)}var n=r;if(n.rank<=1)throw new nF(`batchFlatten requires a minimum rank of 2. Got rank: ${n.rank}.`);let a=[n.shape[0],aI(n.shape,1)];return E.reshape(n,a)})}getConfig(){let e={};return null!=this.dataFormat&&(e.dataFormat=this.dataFormat),Object.assign(e,super.getConfig()),e}}oK.className="Flatten",nJ.registerClass(oK);class oY extends sx{constructor(e){super(e),this.supportsMasking=!0,this.activation=i8(e.activation)}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t);let r=sa(e);return this.activation.apply(r)})}getConfig(){let e={activation:i5(this.activation)};return Object.assign(e,super.getConfig()),e}}oY.className="Activation",nJ.registerClass(oY);class oZ extends sx{constructor(e){super(e),this.n=e.n,this.inputSpec=[{ndim:2}]}computeOutputShape(e){return[e[0],this.n,e[1]]}call(e,t){return(0,n$.tidy)(()=>{var t,r;return t=e=sa(e),r=this.n,(0,n$.tidy)(()=>{if(2!==t.shape.length)throw new nF(`repeat() expects a rank-2 tensor, but received a rank-${t.shape.length} tensor.`);return az(aA(t,1),[1,r,1])})})}getConfig(){let e={n:this.n};return Object.assign(e,super.getConfig()),e}}oZ.className="RepeatVector",nJ.registerClass(oZ);class oJ extends sx{constructor(e){super(e),this.targetShape=e.targetShape;for(let e=0;e<this.targetShape.length;++e)this.isUnknown(this.targetShape[e])&&(this.targetShape[e]=null)}isUnknown(e){return e<0||null==e}fixUnknownDimension(e,t){let r="Total size of new array must be unchanged.",n=t.slice(),a=1,s=null;for(let e=0;e<n.length;++e){let t=n[e];if(this.isUnknown(t))if(null===s)s=e;else throw new nF("Can only specifiy one unknown dimension.");else a*=t}let i=aI(e);if(null!==s){if(0===a||i%a!=0)throw new nF(r);n[s]=i/a}else if(i!==a)throw new nF(r);return n}computeOutputShape(e){let t=!1;for(let r=0;r<e.length;++r)if(this.isUnknown(e[r])){t=!0;break}return t?e.slice(0,1).concat(this.targetShape):e.slice(0,1).concat(this.fixUnknownDimension(e.slice(1),this.targetShape))}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t);let r=sa(e),n=r.shape,a=n.slice(0,1).concat(this.fixUnknownDimension(n.slice(1),this.targetShape));return(0,E.reshape)(r,a)})}getConfig(){let e={targetShape:this.targetShape};return Object.assign(e,super.getConfig()),e}}oJ.className="Reshape",nJ.registerClass(oJ);class oQ extends sx{constructor(e){if(super(e),null==e.dims)throw Error("Required configuration field `dims` is missing during Permute constructor call.");if(!Array.isArray(e.dims))throw Error(`Permute constructor requires \`dims\` to be an Array, but received ${e.dims} instead.`);const t=aT(1,e.dims.length+1);if(!nR.util.arraysEqual(e.dims.slice().sort(),t))throw Error("Invalid permutation `dims`: "+JSON.stringify(e.dims)+" `dims` must contain consecutive integers starting from 1.");this.dims=e.dims,this.dimsIncludingBatch=[0].concat(this.dims),this.inputSpec=[new sp({ndim:this.dims.length+1})]}computeOutputShape(e){let t=(e=ss(e)).slice();return this.dims.forEach((r,n)=>{t[n+1]=e[r]}),t}call(e,t){return(0,e$.transpose)(sa(e),this.dimsIncludingBatch)}getConfig(){let e={dims:this.dims};return Object.assign(e,super.getConfig()),e}}oQ.className="Permute",nJ.registerClass(oQ);class o0 extends sx{constructor(e){super(null==e?{}:e),this.supportsMasking=!0,null!=e?this.maskValue=null==e.maskValue?0:e.maskValue:this.maskValue=0}computeOutputShape(e){return e}getConfig(){let e=super.getConfig(),t={maskValue:this.maskValue};return Object.assign(t,e),t}computeMask(e,t){let r=sa(e);return(0,rb.any)((0,nr.notEqual)(r,this.maskValue),-1)}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t);let r=sa(e),n=(0,rb.any)((0,nr.notEqual)(r,this.maskValue),-1,!0);return(0,b.mul)(r,(0,y.cast)(n,r.dtype))})}}o0.className="Masking",nJ.registerClass(o0);var nJ=nJ;class o1 extends sx{constructor(e){if(super(e),this.embeddings=null,this.DEFAULT_EMBEDDINGS_INITIALIZER="randomUniform",null==e.batchInputShape&&null==e.inputShape){let t=null;null!=e.batchSize&&(t=e.batchSize),null==e.inputLength?this.batchInputShape=[t,null]:this.batchInputShape=[t].concat(nB(e.inputLength))}this.inputDim=e.inputDim,nY(this.inputDim,"inputDim"),this.outputDim=e.outputDim,nY(this.outputDim,"outputDim"),this.embeddingsInitializer=st(e.embeddingsInitializer||this.DEFAULT_EMBEDDINGS_INITIALIZER),this.embeddingsRegularizer=on(e.embeddingsRegularizer),this.activityRegularizer=on(e.activityRegularizer),this.embeddingsConstraint=sD(e.embeddingsConstraint),this.maskZero=e.maskZero,this.supportsMasking=e.maskZero,this.inputLength=e.inputLength}build(e){this.embeddings=this.addWeight("embeddings",[this.inputDim,this.outputDim],this.dtype,this.embeddingsInitializer,this.embeddingsRegularizer,!0,this.embeddingsConstraint),this.built=!0}warnOnIncompatibleInputShape(e){}computeMask(e,t){return(0,n$.tidy)(()=>this.maskZero?(e=sa(e),(0,nr.notEqual)(e,(0,L.zerosLike)(e))):null)}computeOutputShape(e){if(e=ss(e),null==this.inputLength)return[...e,this.outputDim];let t=nB(this.inputLength);if(t.length!==e.length-1)throw new nF(`"inputLength" is ${this.inputLength}, but received input shape has shape ${e}`);{let r=0;for(let n=0;n<t.length;++n){let a=t[n],s=e[n+1];if(null!=a&&null!=s&&a!==s)throw new nF(`"inputLength" is ${this.inputLength}, but received input shape has shape ${e}`);null==a&&(t[r]=s),r++}}return[e[0],...t,this.outputDim]}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t);let r=sa(e);"int32"!==r.dtype&&(r=aR(r,"int32"));let n=aP(this.embeddings.read(),(0,E.reshape)(r,[r.size]));return(0,E.reshape)(n,ss(this.computeOutputShape(r.shape)))})}getConfig(){let e={inputDim:this.inputDim,outputDim:this.outputDim,embeddingsInitializer:nV(this.embeddingsInitializer),embeddingsRegularizer:nV(this.embeddingsRegularizer),activityRegularizer:nV(this.activityRegularizer),embeddingsConstraint:nV(this.embeddingsConstraint),maskZero:this.maskZero,inputLength:this.inputLength};return Object.assign(e,super.getConfig()),e}}o1.className="Embedding",nJ.registerClass(o1);var nJ=nJ;class o2 extends sx{constructor(e){super(e||{}),this.supportsMasking=!0}mergeFunction(e){throw new nD}computeElementwiseOpOutputShape(e,t){if(null==e||null==t)return null;if(e.length<t.length)return this.computeElementwiseOpOutputShape(t,e);if(0===t.length)return e;let r=e.slice(0,e.length-t.length);for(let n=0;n<t.length;++n){let a=e[e.length-t.length+n],s=t[n];if(null==a||null==s||a<0||s<0)r.push(null);else if(1===a)r.push(s);else if(1===s)r.push(a);else{if(a!==s)throw new nF("Operands could not be broadcast together with shapes "+JSON.stringify(e)+" "+JSON.stringify(t));r.push(a)}}return r}build(e){if(Array.isArray(e)&&!Array.isArray(e[0])&&(e=[ss(e)]),e.length<2)throw new nF(`A merge layer should be called on an Array of at least 2 inputs. Got ${e.length} input(s).`);let t=[];for(let r of e)null!=r&&null!==r[0]&&t.push(r[0]);if((t=nj(t)).length>1)throw new nF(`Can not merge tensors with different batch sizes. Got tensors with shapes: ${JSON.stringify(e)}.`);let r=null==e[0]?null:e[0].slice(1);for(let t=1;t<e.length;++t){let n=null==e[t]?null:e[t].slice(1);r=this.computeElementwiseOpOutputShape(r,n)}let n=e.map(e=>e.length);-1===e.indexOf(null)&&1===nj(n).length?this.reshapeRequired=!1:this.reshapeRequired=!0}call(e,t){return(0,n$.tidy)(()=>{if(!this.reshapeRequired)return this.mergeFunction(e);{let t=[],r=e.map(e=>e.rank);if(-1===r.indexOf(null)){let n=aS(r);for(let r of e){let e=r.rank;for(let t=0;t<n-e;++t)r=aA(r,1);t.push(r)}return this.mergeFunction(t)}{let r=!1;for(let n of e){let e=n.rank;if(null==e){let e=n.shape,a=e[0],s=e.slice(1).concat([a]),i=E.reshape(n,[a].concat(aI(e.slice(1))));i=e$.transpose(i,[1,0]),i=E.reshape(i,s),t.push(i),r=!0}else if(e>1){let a=aT(1,e).concat([0]);t.push(e$.transpose(n,a)),r=!0}else t.push(n)}let n=this.mergeFunction(t),a=n.rank;if(r){if(null==a){let e=n.shape,t=e.length,r=e[t-1],a=[r].concat(e.slice(0,e.length-1));n=E.reshape(e$.transpose(E.reshape(n,[-1,r]),[1,0]),a)}else if(a>1){let e=[a-1].concat(aT(0,a-1));n=e$.transpose(n,e)}}return n}}})}computeOutputShape(e){let t;t=null==e[0]?null:e[0].slice(1);for(let r=1;r<e.length;++r){let n=null==e[r]?null:e[r].slice(1);t=this.computeElementwiseOpOutputShape(t,n)}let r=[];for(let t of e)null!=t&&null!==t[0]&&r.push(t[0]);return 1===(r=nj(r)).length?r.concat(t):[null].concat(t)}computeMask(e,t){return n$.tidy(()=>{if(null==t)return null;if(!Array.isArray(t))throw new nF("`mask` should be an Array");if(!Array.isArray(e))throw new nF("`inputs` should be an Array");if(t.length!==e.length)throw new nF(`The Array 'inputs' and 'mask' are expected to have the same length, but have different lengths (${e.length} vs ${t.length})`);if(t.every(e=>null==e))return null;let r=(t=t.map(e=>null==e?e:rl.expandDims(e,0)))[0];for(let e=1;e<t.length-1;++e)r=el.logicalAnd(r,t[e]);return r})}}class o3 extends o2{constructor(e){super(e)}mergeFunction(e){return(0,n$.tidy)(()=>{let t=e[0].clone();for(let r=1;r<e.length;++r)t=P.add(t,e[r]);return t})}}o3.className="Add",nJ.registerClass(o3);class o4 extends o2{constructor(e){super(e)}mergeFunction(e){return(0,n$.tidy)(()=>{let t=e[0].clone();for(let r=1;r<e.length;++r)t=b.mul(t,e[r]);return t})}}o4.className="Multiply",nJ.registerClass(o4);class o5 extends o2{constructor(e){super(e)}mergeFunction(e){return(0,n$.tidy)(()=>{let t=e[0].clone();for(let r=1;r<e.length;++r)t=P.add(t,e[r]);return b.mul(1/e.length,t)})}}o5.className="Average",nJ.registerClass(o5);class o6 extends o2{constructor(e){super(e)}mergeFunction(e){return(0,n$.tidy)(()=>{let t=e[0];for(let r=1;r<e.length;++r)t=rh.maximum(t,e[r]);return t})}}o6.className="Maximum",nJ.registerClass(o6);class o8 extends o2{constructor(e){super(e)}mergeFunction(e){return(0,n$.tidy)(()=>{let t=e[0];for(let r=1;r<e.length;++r)t=r7.minimum(t,e[r]);return t})}}o8.className="Minimum",nJ.registerClass(o8);class o7 extends o2{constructor(e){super(e),this.DEFAULT_AXIS=-1,null==e&&(e={}),this.axis=null==e.axis?this.DEFAULT_AXIS:e.axis,this.supportsMasking=!0,this.reshapeRequired=!1}build(e){if(!(Array.isArray(e)&&Array.isArray(e[0]))||1===e.length)throw new nF("A `Concatenate` layer should be called on a list of at least 2 inputs");let t=!0;for(let r of e)if(null!=r){t=!1;break}if(t)return;let r=[];for(let t=0;t<e.length;++t){let n=e[t].slice();n.splice(this.axis,1);let a=!1;for(let e of r)if(nR.util.arraysEqual(e,n)){a=!0;break}a||r.push(n)}if(r.length>1)throw new nF("A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got input shapes: "+JSON.stringify(e))}mergeFunction(e){return(0,n$.tidy)(()=>aO(e,this.axis))}computeOutputShape(e){if(!(Array.isArray(e)&&Array.isArray(e[0])))throw new nF("A `Concatenate` layer should be called on a list of inputs.");let t=e[0].slice(),r=this.axis<0?t.length+this.axis:this.axis;for(let n of e.slice(1)){if(null==t[r]||null==n[r]){t[r]=null;break}t[r]+=n[r]}return t}computeMask(e,t){if(null==t)return null;if(!Array.isArray(t))throw new nF("`mask` should be an array for Concatenate");if(!Array.isArray(e))throw new nF("`inputs` should be an array for Concatenate");if(t.length!==e.length)throw new nF(`Mismatch in the length of mask (${t.length}) and the legnth of inputs (${e.length})`);return n$.tidy(()=>{let r=!0;if(t.forEach(e=>{if(null!=e){r=!1;return}}),r)return null;let n=[];for(let r=0;r<e.length;++r)null==t[r]?n.push(y.cast(na.onesLike(e[r]),"bool")):t[r].rank<e[r].rank?n.push(rl.expandDims(t[r],-1)):n.push(t[r]);let a=t5.concat(n,this.axis);return ry.all(a,-1,!1)})}getConfig(){let e={axis:this.axis};return Object.assign(e,super.getConfig()),e}}function o9(e,t){for(;e<0;)e+=t;return e}o7.className="Concatenate",nJ.registerClass(o7);class le extends o2{constructor(e){super(e),this.axes=e.axes,this.normalize=null!=e.normalize&&e.normalize,this.supportsMasking=!0,this.reshapeRequired=!1}build(e){nR.util.assert(Array.isArray(e)&&2===e.length&&Array.isArray(e[0])&&Array.isArray(e[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");let t=e[0],r=e[1];if(t.length>3||r.length>3)throw new nD("Dot layer does not support tensors of 4D or higher rank yet.");let n=this.interpretAxes(t,r);if(t[n[0]]!==r[n[1]])throw new nF(`Dimension incompatibility: ${t[n[0]]} !== ${r[n[1]]}`)}mergeFunction(e){let t;if(2!==e.length)throw new nF(`A \`Dot\` layer must be called on exactly 2 inputs, but received ${e.length} input(s).`);let r=e[0],n=e[1];t=Array.isArray(this.axes)?this.axes.map((t,r)=>o9(t,e[r].shape.length)):[o9(this.axes,r.shape.length),o9(this.axes,n.shape.length)],this.normalize&&(r=sZ(r,t[0]),n=sZ(n,t[1]));var a=r,s=n,i=t;if(a.shape.length>3||s.shape.length>3)throw new nD("batchDot is not implemented for tensors of 4D or higher rank yet");if(nR.util.assert(a.shape.length>=2,()=>`batchDot requires the rank of x to be >= 2, but got ${a.shape.length}`),nR.util.assert(a.shape.length>=2,()=>`batchDot requires the rank of y to be >= 2, but got ${s.shape.length}`),"number"==typeof i&&(i=[i,i]),"complex64"===a.dtype||"complex64"===s.dtype)throw new nD("batchDot is not implemented for complex64-type Tensors yet.");let o=a.shape.length,l=s.shape.length;null==i&&(i=[o-1,l-2]);let u=i;return n$.tidy(()=>{let e,t;if(o>l){e=o-l;let t=[];for(let r=0;r<e;++r)t.push(1);s=E.reshape(s,s.shape.concat(t))}else if(l>o){e=l-o;let t=[];for(let r=0;r<e;++r)t.push(1);a=E.reshape(a,a.shape.concat(t))}else e=0;if(2===a.shape.length&&2===s.shape.length)t=u[0]===u[1]?F.sum(b.mul(a,s),u[0]):F.sum(b.mul(e$.transpose(a,[1,0]),s),u[1]);else{let e=u[0]!==a.shape.length-1,r=u[1]===s.shape.length-1;t=Q.matMul(a,s,e,r)}if(e>0){let r;r=o>l?o+l-3:o-1;let n=[];for(let t=r;t<r+e;++t)n.push(t);t=nw.squeeze(t,n)}return 1===t.shape.length&&(t=rl.expandDims(t,1)),t})}interpretAxes(e,t){return Array.isArray(this.axes)?this.axes:[o9(this.axes,e.length),o9(this.axes,t.length)]}computeOutputShape(e){nR.util.assert(Array.isArray(e)&&2===e.length&&Array.isArray(e[0])&&Array.isArray(e[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");let t=e[0].slice(),r=e[1].slice();if(t.length>3||r.length>3)throw new nD("Dot layer does not support tensors of 4D or higher rank yet.");let n=this.interpretAxes(t,r);t.splice(n[0],1),r.splice(n[1],1),r.splice(0,1);let a=t.concat(r);return 1===a.length&&a.push(1),a}computeMask(e,t){return null}getConfig(){let e={axes:this.axes,normalize:this.normalize};return Object.assign(e,super.getConfig()),e}}le.className="Dot",nJ.registerClass(le);var nJ=nJ;class lt extends sx{constructor(e){super(e),this.supportsMasking=!0,this.stddev=e.stddev}computeOutputShape(e){return e}getConfig(){let e=super.getConfig(),t={stddev:this.stddev};return Object.assign(t,e),t}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t);let r=sa(e);return aV(()=>(0,P.add)(a_(r.shape,0,this.stddev),r),()=>r,t.training||!1)})}}lt.className="GaussianNoise",nJ.registerClass(lt);class lr extends sx{constructor(e){super(e),this.supportsMasking=!0,this.rate=e.rate}computeOutputShape(e){return e}getConfig(){let e=super.getConfig(),t={rate:this.rate};return Object.assign(t,e),t}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t);let r=sa(e);return this.rate>0&&this.rate<1?aV(()=>{let e=Math.sqrt(this.rate/(1-this.rate));return(0,b.mul)(r,a_(r.shape,1,e))},()=>r,t.training||!1):r})}}lr.className="GaussianDropout",nJ.registerClass(lr);class ln extends sx{constructor(e){super(e),this.supportsMasking=!0,this.rate=e.rate,this.noiseShape=e.noiseShape}_getNoiseShape(e){return this.noiseShape||sa(e).shape}computeOutputShape(e){return e}getConfig(){let e=super.getConfig(),t={rate:this.rate};return Object.assign(t,e),t}call(e,t){return(0,n$.tidy)(()=>{if(this.rate<1&&this.rate>0){let r=this._getNoiseShape(e);return aV(()=>{let t=sa(e),n=(0,ei.greaterEqual)((0,au.randomUniform)(r),this.rate);n=aR(n,"float32");let a=((1-this.rate)*(1+3.09091329228798*this.rate))**-.5,s=-(-1.7580993408473766*a)*this.rate,i=(0,P.add)((0,b.mul)(t,n),(0,b.mul)((0,P.add)(n,-1),-1.7580993408473766));return(0,P.add)((0,b.mul)(i,a),s)},()=>sa(e),t.training||!1)}return e})}}ln.className="AlphaDropout",nJ.registerClass(ln);var la=e.i(28808),ls=e.i(26213),li=e.i(77760),lo=e.i(91949),nJ=nJ;function ll(e,t,r,n,a,s=.001){let i;if(2===e.rank)i=li.batchNorm2d(e,t,r,n,a,s);else if(3===e.rank)i=ls.batchNorm3d(e,t,r,n,a,s);else if(4===e.rank)i=la.batchNorm4d(e,t,r,n,a,s);else throw new nD(`batchNormalization is not implemented for array of rank ${e.rank} yet`);return i}class lu extends sx{constructor(e){null==e&&(e={}),super(e),this.supportsMasking=!0,this.axis=null==e.axis?-1:e.axis,this.momentum=null==e.momentum?.99:e.momentum,this.epsilon=null==e.epsilon?.001:e.epsilon,this.center=null==e.center||e.center,this.scale=null==e.scale||e.scale,this.betaInitializer=st(e.betaInitializer||"zeros"),this.gammaInitializer=st(e.gammaInitializer||"ones"),this.movingMeanInitializer=st(e.movingMeanInitializer||"zeros"),this.movingVarianceInitializer=st(e.movingVarianceInitializer||"ones"),this.betaConstraint=sD(e.betaConstraint),this.gammaConstraint=sD(e.gammaConstraint),this.betaRegularizer=on(e.betaRegularizer),this.gammaRegularizer=on(e.gammaRegularizer)}build(e){e=ss(e);let t=this.axis>=0?this.axis:this.axis+e.length,r=e[t];if(null==r)throw new nF(`Axis ${t} of input tensor should have a defined dimension but the layer received an input with shape ${JSON.stringify(e)}.`);this.inputSpec=[new sp({ndim:e.length,axes:{[t]:r}})];let n=[r];this.scale&&(this.gamma=this.addWeight("gamma",n,null,this.gammaInitializer,this.gammaRegularizer,!0,this.gammaConstraint)),this.center&&(this.beta=this.addWeight("beta",n,null,this.betaInitializer,this.betaRegularizer,!0,this.betaConstraint)),this.movingMean=this.addWeight("moving_mean",n,null,this.movingMeanInitializer,null,!1),this.movingVariance=this.addWeight("moving_variance",n,null,this.movingVarianceInitializer,null,!1),this.built=!0}call(e,t){return(0,n$.tidy)(()=>{let r=null!=t.training&&t.training,n=sa(e),a=n.shape,s=a.length,i=aT(0,s),o=this.axis>=0?this.axis:this.axis+s;i.splice(o,1);let l=nz(1,s);l[o]=a[o];let u=i.slice();u.sort();let h=!nR.util.arraysEqual(u,aT(0,s).slice(0,s-1)),d=()=>h?ll(n,(0,E.reshape)(this.movingMean.read(),l),(0,E.reshape)(this.movingVariance.read(),l),this.center?(0,E.reshape)(this.beta.read(),l):null,this.scale?(0,E.reshape)(this.gamma.read(),l):null,this.epsilon):ll(n,this.movingMean.read(),this.movingVariance.read(),null==this.beta?null:this.beta.read(),null==this.gamma?null:this.gamma.read(),this.epsilon);if(!r)return d();let[p,c,f]=function(e,t,r,n,a=.001){return nR.util.arraysEqual(n.slice().sort(),aT(0,e.rank-1))?function(e,t,r,n,a=.001){return(0,n$.tidy)(()=>{let s=lo.moments(e,n),i=s.mean,o=s.variance;return[ll(e,i,o,r,t,a),i,o]})}(e,t,r,n,a):function(e,t,r,n,a=.001){return(0,n$.tidy)(()=>{let s=lo.moments(e,n),i=s.mean,o=s.variance,l=[];for(let t of aT(0,e.rank))-1!==n.indexOf(t)?l.push(1):l.push(e.shape[t]);let u=(0,E.reshape)(i,l),h=(0,E.reshape)(o,l),d=null==t?null:(0,E.reshape)(t,l);return[ll(e,u,h,null==r?null:(0,E.reshape)(r,l),d,a),i,o]})}(e,t,r,n,a)}(n,this.gamma.read(),this.beta.read(),i,this.epsilon),m=(e,t,r)=>{n$.tidy(()=>{let n=e.read(),a=b.mul(N.sub(n,t),1-r);e.write(N.sub(n,a))})};return m(this.movingMean,c,this.momentum),m(this.movingVariance,f,this.momentum),p})}getConfig(){let e={axis:this.axis,momentum:this.momentum,epsilon:this.epsilon,center:this.center,scale:this.scale,betaInitializer:nV(this.betaInitializer),gammaInitializer:nV(this.gammaInitializer),movingMeanInitializer:nV(this.movingMeanInitializer),movingVarianceInitializer:nV(this.movingVarianceInitializer),betaRegularizer:nV(this.betaRegularizer),gammaRegularizer:nV(this.gammaRegularizer),betaConstraint:nV(this.betaConstraint),gammaConstraint:nV(this.gammaConstraint)};return Object.assign(e,super.getConfig()),e}}lu.className="BatchNormalization",nJ.registerClass(lu);class lh extends sx{constructor(e){if(null==e&&(e={}),super(e),this.axis=null==e.axis?-1:e.axis,"number"==typeof this.axis){if(!Number.isInteger(this.axis))throw Error(`Expected axis to be an integer, but received ${this.axis}`)}else if(Array.isArray(this.axis)){for(const e of this.axis)if(!Number.isInteger(e))throw Error(`Expected axis to be an array of integers, but received ${JSON.stringify(this.axis)}`)}else throw Error(`Expected axis to be an integer or an array of integers, but received ${JSON.stringify(this.axis)}`);this.epsilon=null==e.epsilon?.001:e.epsilon,this.center=null==e.center||e.center,this.scale=null==e.scale||e.scale,this.betaInitializer=st(e.betaInitializer||"zeros"),this.gammaInitializer=st(e.gammaInitializer||"ones"),this.betaRegularizer=on(e.betaRegularizer),this.gammaRegularizer=on(e.gammaRegularizer),this.supportsMasking=!0}build(e){let t=(e=ss(e)).length;"number"==typeof this.axis&&(this.axis=[this.axis]);for(let e=0;e<this.axis.length;++e)this.axis[e]<0&&(this.axis[e]+=t);for(let e of this.axis)if(e<0||e>=t)throw Error(`Invalid axis: ${e}`);if(this.axis.length!==nj(this.axis).length)throw Error(`Found duplicate axes in: ${this.axis}`);let r=this.axis.map(t=>e[t]);this.scale?this.gamma=this.addWeight("gamma",r,"float32",this.gammaInitializer,this.gammaRegularizer,!0):this.gamma=null,this.center?this.beta=this.addWeight("beta",r,"float32",this.betaInitializer,this.betaRegularizer,!0):this.beta=null,this.built=!0}call(e,t){let r=sa(e),n=r.shape,a=n.length;return(0,n$.tidy)(()=>{let{mean:e,variance:t}=(0,lo.moments)(r,this.axis,!0),s=nz(1,a);for(let e of this.axis)s[e]=n[e];let i=e=>null!=e&&e.shape.length!==a?E.reshape(e,s):e,o=this.scale?i(this.gamma.read()):null,l=this.center?i(this.beta.read()):null,u=[],h=[];for(let e=0;e<a;++e)-1!==this.axis.indexOf(e)?(u.push(n[e]),h.push(1)):(u.push(1),h.push(n[e]));return e=eU.tile(e,u),t=eU.tile(t,u),null!=o&&(o=eU.tile(o,h)),null!=l&&(l=eU.tile(l,h)),ll(r,e,t,l,o,this.epsilon)})}getConfig(){let e={axis:this.axis,epsilon:this.epsilon,center:this.center,scale:this.scale,betaInitializer:nV(this.betaInitializer),gammaInitializer:nV(this.gammaInitializer),betaRegularizer:nV(this.betaRegularizer),gammaRegularizer:nV(this.gammaRegularizer)};return Object.assign(e,super.getConfig()),e}}lh.className="LayerNormalization",nJ.registerClass(lh);var nJ=nJ;class ld extends sx{constructor(e){if(null==e&&(e={}),super(e),this.dataFormat=null==e.dataFormat?a$():e.dataFormat,null==e.padding)this.padding=[[1,1],[1,1]];else if("number"==typeof e.padding)this.padding=[[e.padding,e.padding],[e.padding,e.padding]];else{let t,r;if(e.padding=e.padding,2!==e.padding.length)throw new nF(`ZeroPadding2D expects padding to be a length-2 array, but received a length-${e.padding.length} array.`);if("number"==typeof e.padding[0])t=[e.padding[0],e.padding[0]],r=[e.padding[1],e.padding[1]];else{if(e.padding=e.padding,2!==e.padding[0].length)throw new nF(`ZeroPadding2D expects height padding to be a length-2 array, but received a length-${e.padding[0].length} array.`);if(t=e.padding[0],2!==e.padding[1].length)throw new nF(`ZeroPadding2D expects width padding to be a length-2 array, but received a length-${e.padding[1].length} array.`);r=e.padding[1]}this.padding=[t,r]}this.inputSpec=[new sp({ndim:4})]}computeOutputShape(e){let t,r;return(e=ss(e),"channelsFirst"===this.dataFormat)?(t=null!=e[2]&&e[2]>=0?e[2]+this.padding[0][0]+this.padding[0][1]:null,r=null!=e[3]&&e[3]>=0?e[3]+this.padding[1][0]+this.padding[1][1]:null,[e[0],e[1],t,r]):(t=null!=e[1]&&e[1]>=0?e[1]+this.padding[0][0]+this.padding[0][1]:null,r=null!=e[2]&&e[2]>=0?e[2]+this.padding[1][0]+this.padding[1][1]:null,[e[0],t,r,e[3]])}call(e,t){return(0,n$.tidy)(()=>{var t,r,n;return t=sa(e),r=this.padding,n=this.dataFormat,(0,n$.tidy)(()=>{let e;if(4!==t.rank)throw new nF(`temporalPadding expects input tensor to be 4-D, but received a ${t.rank}-D tensor.`);if(null==r&&(r=[[1,1],[1,1]]),2!==r.length||2!==r[0].length||2!==r[1].length)throw new nF("spatial2dPadding expects `padding` to be an Array of two Arrays, each of which is an Array of two integers.");if(null==n&&(n=a$()),"channelsLast"!==n&&"channelsFirst"!==n)throw new nF(`Unknown data format: ${n}. Supported data formats are 'channelsLast' and 'channelsFirst.`);return e="channelsFirst"===n?[[0,0],[0,0],r[0],r[1]]:[[0,0],r[0],r[1],[0,0]],tZ.pad(t,e)})})}getConfig(){let e={padding:this.padding,dataFormat:this.dataFormat};return Object.assign(e,super.getConfig()),e}}ld.className="ZeroPadding2D",nJ.registerClass(ld);var lp=e.i(69705),lc=e.i(51158),nJ=nJ;function lf(e,t,r,n,a,s){return(0,n$.tidy)(()=>{let i;n7(a),ae(s),n9(n),null==r&&(r=[1,1]),null==n&&(n="valid"),null==a&&(a=a$()),null==s&&(s="max"),e=om(e,a);let o="same"===n?"same":"valid";return i="max"===s?r4.maxPool(e,t,r,o):rN.avgPool(e,t,r,o),"channelsFirst"===a&&(i=e$.transpose(i,[0,3,1,2])),i})}function lm(e,t,r,n,a,s){return(0,n$.tidy)(()=>{let i;n7(a),ae(s),n9(n),null==r&&(r=[1,1,1]),null==n&&(n="valid"),null==a&&(a=a$()),null==s&&(s="max"),e=og(e,a);let o="same"===n?"same":"valid";return i="max"===s?lc.maxPool3d(e,t,r,o):lp.avgPool3d(e,t,r,o),"channelsFirst"===a&&(i=e$.transpose(i,[0,4,1,2,3])),i})}class lg extends sx{constructor(e){if(null==e.poolSize&&(e.poolSize=2),super(e),"number"==typeof e.poolSize)this.poolSize=[e.poolSize];else if(Array.isArray(e.poolSize)&&1===e.poolSize.length&&"number"==typeof e.poolSize[0])this.poolSize=e.poolSize;else throw new nF(`poolSize for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(e.poolSize)}`);if(nY(this.poolSize,"poolSize"),null==e.strides)this.strides=this.poolSize;else if("number"==typeof e.strides)this.strides=[e.strides];else if(Array.isArray(e.strides)&&1===e.strides.length&&"number"==typeof e.strides[0])this.strides=e.strides;else throw new nF(`strides for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(e.strides)}`);nY(this.strides,"strides"),this.padding=null==e.padding?"valid":e.padding,n9(this.padding),this.inputSpec=[new sp({ndim:3})]}computeOutputShape(e){let t=oc((e=ss(e))[1],this.poolSize[0],this.padding,this.strides[0]);return[e[0],t,e[2]]}call(e,t){return(0,n$.tidy)(()=>{this.invokeCallHook(e,t),e=aA(sa(e),2);let r=this.poolingFunction(sa(e),[this.poolSize[0],1],[this.strides[0],1],this.padding,"channelsLast");return nw.squeeze(r,[2])})}getConfig(){let e={poolSize:this.poolSize,padding:this.padding,strides:this.strides};return Object.assign(e,super.getConfig()),e}}class lx extends lg{constructor(e){super(e)}poolingFunction(e,t,r,n,a){return n7(a),n9(n),lf(e,t,r,n,a,"max")}}lx.className="MaxPooling1D",nJ.registerClass(lx);class ly extends lg{constructor(e){super(e)}poolingFunction(e,t,r,n,a){return n7(a),n9(n),lf(e,t,r,n,a,"avg")}}ly.className="AveragePooling1D",nJ.registerClass(ly);class lb extends sx{constructor(e){if(null==e.poolSize&&(e.poolSize=[2,2]),super(e),this.poolSize=Array.isArray(e.poolSize)?e.poolSize:[e.poolSize,e.poolSize],null==e.strides)this.strides=this.poolSize;else if(Array.isArray(e.strides)){if(2!==e.strides.length)throw new nF(`If the strides property of a 2D pooling layer is an Array, it is expected to have a length of 2, but received length ${e.strides.length}.`);this.strides=e.strides}else this.strides=[e.strides,e.strides];nY(this.poolSize,"poolSize"),nY(this.strides,"strides"),this.padding=null==e.padding?"valid":e.padding,this.dataFormat=null==e.dataFormat?"channelsLast":e.dataFormat,n7(this.dataFormat),n9(this.padding),this.inputSpec=[new sp({ndim:4})]}computeOutputShape(e){e=ss(e);let t="channelsFirst"===this.dataFormat?e[2]:e[1],r="channelsFirst"===this.dataFormat?e[3]:e[2];return(t=oc(t,this.poolSize[0],this.padding,this.strides[0]),r=oc(r,this.poolSize[1],this.padding,this.strides[1]),"channelsFirst"===this.dataFormat)?[e[0],e[1],t,r]:[e[0],t,r,e[3]]}call(e,t){return(0,n$.tidy)(()=>(this.invokeCallHook(e,t),this.poolingFunction(sa(e),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){let e={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat};return Object.assign(e,super.getConfig()),e}}class lv extends lb{constructor(e){super(e)}poolingFunction(e,t,r,n,a){return n7(a),n9(n),lf(e,t,r,n,a,"max")}}lv.className="MaxPooling2D",nJ.registerClass(lv);class lw extends lb{constructor(e){super(e)}poolingFunction(e,t,r,n,a){return n7(a),n9(n),lf(e,t,r,n,a,"avg")}}lw.className="AveragePooling2D",nJ.registerClass(lw);class lC extends sx{constructor(e){if(null==e.poolSize&&(e.poolSize=[2,2,2]),super(e),this.poolSize=Array.isArray(e.poolSize)?e.poolSize:[e.poolSize,e.poolSize,e.poolSize],null==e.strides)this.strides=this.poolSize;else if(Array.isArray(e.strides)){if(3!==e.strides.length)throw new nF(`If the strides property of a 3D pooling layer is an Array, it is expected to have a length of 3, but received length ${e.strides.length}.`);this.strides=e.strides}else this.strides=[e.strides,e.strides,e.strides];nY(this.poolSize,"poolSize"),nY(this.strides,"strides"),this.padding=null==e.padding?"valid":e.padding,this.dataFormat=null==e.dataFormat?"channelsLast":e.dataFormat,n7(this.dataFormat),n9(this.padding),this.inputSpec=[new sp({ndim:5})]}computeOutputShape(e){e=ss(e);let t="channelsFirst"===this.dataFormat?e[2]:e[1],r="channelsFirst"===this.dataFormat?e[3]:e[2],n="channelsFirst"===this.dataFormat?e[4]:e[3];return(t=oc(t,this.poolSize[0],this.padding,this.strides[0]),r=oc(r,this.poolSize[1],this.padding,this.strides[1]),n=oc(n,this.poolSize[2],this.padding,this.strides[2]),"channelsFirst"===this.dataFormat)?[e[0],e[1],t,r,n]:[e[0],t,r,n,e[4]]}call(e,t){return(0,n$.tidy)(()=>(this.invokeCallHook(e,t),this.poolingFunction(sa(e),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){let e={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat};return Object.assign(e,super.getConfig()),e}}class lI extends lC{constructor(e){super(e)}poolingFunction(e,t,r,n,a){return n7(a),n9(n),lm(e,t,r,n,a,"max")}}lI.className="MaxPooling3D",nJ.registerClass(lI);class lk extends lC{constructor(e){super(e)}poolingFunction(e,t,r,n,a){return n7(a),n9(n),lm(e,t,r,n,a,"avg")}}lk.className="AveragePooling3D",nJ.registerClass(lk);class lS extends sx{constructor(e){super(e),this.inputSpec=[new sp({ndim:3})]}computeOutputShape(e){return[e[0],e[2]]}call(e,t){throw new nD}}class lT extends lS{constructor(e){super(e||{})}call(e,t){return(0,n$.tidy)(()=>{let t=sa(e);return r6.mean(t,1)})}}lT.className="GlobalAveragePooling1D",nJ.registerClass(lT);class lN extends lS{constructor(e){super(e||{})}call(e,t){return(0,n$.tidy)(()=>{let t=sa(e);return r5.max(t,1)})}}lN.className="GlobalMaxPooling1D",nJ.registerClass(lN);class l$ extends sx{constructor(e){super(e),this.dataFormat=null==e.dataFormat?"channelsLast":e.dataFormat,n7(this.dataFormat),this.inputSpec=[new sp({ndim:4})]}computeOutputShape(e){return"channelsLast"===this.dataFormat?[e[0],e[3]]:[e[0],e[1]]}call(e,t){throw new nD}getConfig(){let e={dataFormat:this.dataFormat};return Object.assign(e,super.getConfig()),e}}class lR extends l${call(e,t){return(0,n$.tidy)(()=>{let t=sa(e);return"channelsLast"===this.dataFormat?r6.mean(t,[1,2]):r6.mean(t,[2,3])})}}lR.className="GlobalAveragePooling2D",nJ.registerClass(lR);class lA extends l${call(e,t){return(0,n$.tidy)(()=>{let t=sa(e);return"channelsLast"===this.dataFormat?r5.max(t,[1,2]):r5.max(t,[2,3])})}}lA.className="GlobalMaxPooling2D",nJ.registerClass(lA);var nJ=nJ;class lE extends sx{constructor(e){super(e),this.layer=e.layer}build(e){this.built=!0}get trainable(){return null!=this.layer&&this.layer.trainable}set trainable(e){null!=this.layer&&(this.layer.trainable=e)}get trainableWeights(){return this.layer.trainableWeights}get nonTrainableWeights(){return this.layer.nonTrainableWeights}get updates(){return this.layer._updates}get losses(){return this.layer.losses}getWeights(){return this.layer.getWeights()}setWeights(e){this.layer.setWeights(e)}getConfig(){let e={layer:{className:this.layer.getClassName(),config:this.layer.getConfig()}};return Object.assign(e,super.getConfig()),e}setFastWeightInitDuringBuild(e){super.setFastWeightInitDuringBuild(e),null!=this.layer&&this.layer.setFastWeightInitDuringBuild(e)}static fromConfig(e,t,r={}){let n=sK(t.layer,r);delete t.layer;let a={layer:n};return Object.assign(a,t),new e(a)}}class lF extends lE{constructor(e){super(e),this.supportsMasking=!0}build(e){if((e=ss(e)).length<3)throw new nF(`TimeDistributed layer expects an input shape >= 3D, but received input shape ${JSON.stringify(e)}`);this.inputSpec=[{shape:e}];let t=[e[0]].concat(e.slice(2));this.layer.built||(this.layer.build(t),this.layer.built=!0),super.build(e)}computeOutputShape(e){let t=[(e=ss(e))[0]].concat(e.slice(2)),r=this.layer.computeOutputShape(t),n=e[1];return[r[0],n].concat(r.slice(1))}call(e,t){return(0,n$.tidy)(()=>oE((e,r)=>[sa(this.layer.call(e,t)),[]],e=sa(e),[],!1,null,null,!1,!0)[1])}}lF.className="TimeDistributed",nJ.registerClass(lF);class lD extends lE{constructor(e){super(e);const t=e.layer.getConfig(),r={};r.className=e.layer.getClassName(),r.config=t,this.forwardLayer=sK(r),t.goBackwards=!0!==t.goBackwards;const n={};if(n.className=e.layer.getClassName(),n.config=t,this.backwardLayer=sK(n),this.forwardLayer.name="forward_"+this.forwardLayer.name,this.backwardLayer.name="backward_"+this.backwardLayer.name,this.mergeMode=void 0===e.mergeMode?"concat":e.mergeMode,!function(e){nX(n6,"BidirectionalMergeMode",e)}(this.mergeMode),e.weights)throw new nD("weights support is not implemented for Bidirectional layer yet.");this._stateful=e.layer.stateful,this.returnSequences=e.layer.returnSequences,this.returnState=e.layer.returnState,this.supportsMasking=!0,this._trainable=!0,this.inputSpec=e.layer.inputSpec,this.numConstants=null}get trainable(){return this._trainable}set trainable(e){this._trainable=e,null!=this.forwardLayer&&(this.forwardLayer.trainable=e),null!=this.backwardLayer&&(this.backwardLayer.trainable=e)}getWeights(){return this.forwardLayer.getWeights().concat(this.backwardLayer.getWeights())}setWeights(e){let t=Math.floor(e.length/2);this.forwardLayer.setWeights(e.slice(0,t)),this.backwardLayer.setWeights(e.slice(t))}computeOutputShape(e){let t,r,n,a=this.forwardLayer.computeOutputShape(e);return(Array.isArray(a)&&Array.isArray(a[0])||(a=[a]),this.returnState&&(n=a.slice(1)),t=a[0],"concat"===this.mergeMode?(t[t.length-1]*=2,r=[t]):r=null==this.mergeMode?[t,t.slice()]:[t],this.returnState)?null==this.mergeMode?r.concat(n).concat(n.slice()):[t].concat(n).concat(n.slice()):nP(r)}apply(e,t){let r=null==t?null:t.initialState,n=null==t?null:t.constants;null==t&&(t={});let a=oA(e,r,n,this.numConstants);if(e=a.inputs,r=a.initialState,n=a.constants,Array.isArray(e)&&(r=e.slice(1),e=e[0]),(null==r||0===r.length)&&null==n)return super.apply(e,t);let s=[],i=[];if(null!=r){let e=r.length;if(e%2>0)throw new nF("When passing `initialState` to a Bidrectional RNN, the state should be an Array containing the states of the underlying RNNs.");t.initialState=r,s.push(...r);let n=r.map(e=>new sp({shape:e.shape}));this.forwardLayer.stateSpec=n.slice(0,e/2),this.backwardLayer.stateSpec=n.slice(e/2),i.push(...n)}if(null!=n)throw new nD("Support for constants in Bidirectional layers is not implemented yet.");let o=s[0]instanceof sc;for(let e of s)if(e instanceof sc!==o)throw new nF("The initial state of a Bidirectional layer cannot be specified as a mix of symbolic and non-symbolic tensors");if(!o)return super.apply(e,t);{let r=[e].concat(s),n=this.inputSpec.concat(i),a=this.inputSpec;this.inputSpec=n;let o=super.apply(r,t);return this.inputSpec=a,o}}call(e,t){return(0,n$.tidy)(()=>{let r,n,a,s,i=t.initialState;if(null==i)r=this.forwardLayer.call(e,t),n=this.backwardLayer.call(e,t);else{let a=i.slice(0,i.length/2),s=i.slice(i.length/2);r=this.forwardLayer.call(e,Object.assign(t,{initialState:a})),n=this.backwardLayer.call(e,Object.assign(t,{initialState:s}))}return(this.returnState&&(Array.isArray(r)&&(a=r.slice(1).concat(n.slice(1))),r=r[0],n=n[0]),this.returnSequences&&(n=t_.reverse(n,1)),"concat"===this.mergeMode?s=aO([r,n]):"sum"===this.mergeMode?s=P.add(r,n):"ave"===this.mergeMode?s=b.mul(.5,P.add(r,n)):"mul"===this.mergeMode?s=b.mul(r,n):null==this.mergeMode&&(s=[r,n]),this.returnState)?null==this.mergeMode?s.concat(a):[s].concat(a):s})}resetStates(e){this.forwardLayer.resetStates(),this.backwardLayer.resetStates()}build(e){ar(this.forwardLayer.name,()=>{this.forwardLayer.build(e)}),ar(this.backwardLayer.name,()=>{this.backwardLayer.build(e)}),this.built=!0}computeMask(e,t){let r;if(Array.isArray(t)&&(t=t[0]),r=this.returnSequences?null==this.mergeMode?[t,t]:t:null==this.mergeMode?[null,null]:null,!this.returnState)return r;{let e=this.forwardLayer.states.map(e=>null);return Array.isArray(r)?r.concat(e).concat(e):[r].concat(e).concat(e)}}get trainableWeights(){return this.forwardLayer.trainableWeights.concat(this.backwardLayer.trainableWeights)}get nonTrainableWeights(){return this.forwardLayer.nonTrainableWeights.concat(this.backwardLayer.nonTrainableWeights)}setFastWeightInitDuringBuild(e){super.setFastWeightInitDuringBuild(e),null!=this.forwardLayer&&this.forwardLayer.setFastWeightInitDuringBuild(e),null!=this.backwardLayer&&this.backwardLayer.setFastWeightInitDuringBuild(e)}getConfig(){let e={mergeMode:this.mergeMode};return Object.assign(e,super.getConfig()),e}static fromConfig(e,t){let r=sK(t.layer);if(delete t.layer,null!=t.numConstants)throw new nD("Deserialization of a Bidirectional layer with numConstants present is not supported yet.");return t.layer=r,new e(t)}}lD.className="Bidirectional",nJ.registerClass(lD);var nJ=nJ;class lO extends sx{constructor(e){super(e),this.scale=e.scale,e.offset?this.offset=e.offset:this.offset=0}getConfig(){let e={scale:this.scale,offset:this.offset};return Object.assign(e,super.getConfig()),e}call(e,t){return(0,n$.tidy)(()=>("float32"!==(e=sa(e)).dtype&&(e=aR(e,"float32")),(0,P.add)((0,b.mul)(e,this.scale),this.offset)))}}lO.className="Rescaling",nJ.registerClass(lO);var nJ=nJ,lL=e.i(84741),lz=e.i(43616);let{resizeBilinear:l_,cropAndResize:lM}=al.image;class lP extends sx{constructor(e){super(e),this.height=e.height,this.width=e.width}centerCrop(e,t,r,n,a,s,i,o){return(0,n$.tidy)(()=>{let l,u=!1,h=[t/s,r/i,(n+t)/s,(a+r)/i],d=[];3===e.rank?(u=!0,l=(0,eH.stack)([e])):l=e;for(let e=0;e<l.shape[0];e++)d.push(h);let p=lM(l,(0,lL.tensor)(d,[d.length,4]),(0,lz.range)(0,d.length,1,"int32"),[n,a],"nearest");return u?aR(sa((0,tv.unstack)(p)),o):aR(p,o)})}upsize(e,t,r,n){return(0,n$.tidy)(()=>aR(l_(e,[t,r]),n))}call(e,t){return(0,n$.tidy)(()=>{let t=sa(e),r=t.dtype,n=t.shape,a=n[n.length-3],s=n[n.length-2],i=0;a!==this.height&&(i=Math.floor((a-this.height)/2));let o=0;return(s!==this.width&&0===(o=Math.floor((s-this.width)/2))&&(o=1),i>=0&&o>=0)?this.centerCrop(t,i,o,this.height,this.width,a,s,r):this.upsize(e,this.height,this.width,r)})}getConfig(){let e={height:this.height,width:this.width};return Object.assign(e,super.getConfig()),e}computeOutputShape(e){let t=(e=ss(e)).length-3,r=e.length-2;return e[t]=this.height,e[r]=this.width,e}}lP.className="CenterCrop",nJ.registerClass(lP);var nJ=nJ,lB=e.i(9076);class lW extends sx{constructor(e){super(e),this.numTokens=e.numTokens,e.outputMode?this.outputMode=e.outputMode:this.outputMode="multiHot"}getConfig(){let e={numTokens:this.numTokens,outputMode:this.outputMode};return Object.assign(e,super.getConfig()),e}computeOutputShape(e){return null==(e=ss(e))?[this.numTokens]:("oneHot"===this.outputMode&&1!==e[e.length-1]?e.push(this.numTokens):e[e.length-1]=this.numTokens,e)}call(e,t){return(0,n$.tidy)(()=>{let r;if("int32"!==(e=sa(e)).dtype&&(e=aR(e,"int32")),void 0!==t.countWeights){if("count"!==this.outputMode)throw new nF(`countWeights is not used when outputMode !== count.
              Received countWeights=${t.countWeights}`);r=sa(t.countWeights)}let n=(0,r5.max)(e),a=(0,r8.min)(e),s=(0,e1.greater)(this.numTokens,n).bufferSync().get(0),i=(0,ei.greaterEqual)(a,0).bufferSync().get(0);if(!(s&&i))throw new nF(`Input values must be between 0 < values <= numTokens with numTokens=${this.numTokens}`);return function(e,t,r,n){let a,s=sa(e);if("int32"!==s.dtype&&(s=aR(s,"int32")),"int"===t)return s;let i=s.shape;if(0===s.rank&&(s=(0,rl.expandDims)(s,-1)),"oneHot"===t&&1!==s.shape[s.shape.length-1]&&(s=(0,rl.expandDims)(s,-1)),s.rank>2)throw new nF(`When outputMode is not int, maximum output rank is 2 Received outputMode ${t} and input shape ${i} which would result in output rank ${s.rank}.`);let o=["multiHot","oneHot"].includes(t),l=s;if(a=void 0!==n&&"count"===t?(0,lB.denseBincount)(l,n,r,o):(0,lB.denseBincount)(l,[],r,o),"tfIdf"!==t)return a;if(n)return(0,b.mul)(a,n);throw new nF("When outputMode is 'tfIdf', weights must be provided.")}(e,this.outputMode,this.numTokens,r)})}}lW.className="CategoryEncoding",nJ.registerClass(lW);var nJ=nJ;let lG=new Set(["bilinear","nearest"]);class lU extends sx{constructor(e){if(super(e),this.height=e.height,this.width=e.width,e.interpolation)if(lG.has(e.interpolation))this.interpolation=e.interpolation;else throw new nF(`Invalid interpolation parameter: ${e.interpolation} is not implemented`);else this.interpolation="bilinear";this.cropToAspectRatio=!!e.cropToAspectRatio}computeOutputShape(e){let t=(e=ss(e))[2];return[this.height,this.width,t]}getConfig(){let e={height:this.height,width:this.width,interpolation:this.interpolation,cropToAspectRatio:this.cropToAspectRatio};return Object.assign(e,super.getConfig()),e}call(e,t){return(0,n$.tidy)(()=>{let t=[this.height,this.width];if("bilinear"===this.interpolation)return al.image.resizeBilinear(e,t,!this.cropToAspectRatio);if("nearest"===this.interpolation)return al.image.resizeNearestNeighbor(e,t,!this.cropToAspectRatio);throw Error(`Interpolation is ${this.interpolation} but only ${[...lG]} are supported`)})}}lU.className="Resizing",nJ.registerClass(lU);var nJ=nJ;class lV{constructor(e){this.seed=e}next(){if(void 0!==this.seed)return this.seed++}}lV.className="RandomSeed";class lH extends sx{constructor(e){super(e),this.randomGenerator=new lV(e.seed)}getConfig(){let e={seed:this.randomGenerator.seed};return Object.assign(e,super.getConfig()),e}}lH.className="BaseRandomLayer";let lq=new Set(["bilinear","nearest"]);class lj extends lH{constructor(e){super(e);const{factor:t,interpolation:r="bilinear"}=e;if(this.factor=t,Array.isArray(this.factor)&&2===this.factor.length)this.widthLower=this.factor[0],this.widthUpper=this.factor[1];else if(!Array.isArray(this.factor)&&this.factor>0)this.widthLower=-this.factor,this.widthUpper=this.factor;else throw new nF(`Invalid factor: ${this.factor}. Must be positive number or tuple of 2 numbers`);if(this.widthLower<-1||this.widthUpper<-1)throw new nF(`factor must have values larger than -1. Got: ${this.factor}`);if(this.widthUpper<this.widthLower)throw new nF(`factor cannot have upper bound less than lower bound.
        Got upper bound: ${this.widthUpper}.
        Got lower bound: ${this.widthLower}
      `);if(r)if(lq.has(r))this.interpolation=r;else throw new nF(`Invalid interpolation parameter: ${r} is not implemented`)}getConfig(){let e={factor:this.factor,interpolation:this.interpolation};return Object.assign(e,super.getConfig()),e}computeOutputShape(e){let t=(e=ss(e))[2];return[this.imgHeight,-1,t]}call(e,t){return(0,n$.tidy)(()=>{let t=sa(e);this.imgHeight=t.shape[t.shape.length-3];let r=t.shape[t.shape.length-2];this.widthFactor=(0,au.randomUniform)([1],1+this.widthLower,1+this.widthUpper,"float32",this.randomGenerator.next());let n=this.widthFactor.dataSync()[0]*r;n=Math.round(n);let a=[this.imgHeight,n];switch(this.interpolation){case"bilinear":return al.image.resizeBilinear(e,a);case"nearest":return al.image.resizeNearestNeighbor(e,a);default:throw Error(`Interpolation is ${this.interpolation}
          but only ${[...lq]} are supported`)}})}}function lX(e){return new sy(e)}function lK(e){return new oo(e)}function lY(e){return new oa(e)}function lZ(e){return new os(e)}function lJ(e){return new oi(e)}function lQ(e){return new ou(e)}function l0(e){return new ol(e)}function l1(e){return new oT(e)}function l2(e){return new ov(e)}function l3(e){return new oC(e)}function l4(e){return new ow(e)}function l5(e){return new oI(e)}function l6(e){return new oS(e)}function l8(e){return new oN(e)}function l7(e){return new o$(e)}function l9(e){return new oR(e)}function ue(e){return new oY(e)}function ut(e){return new oX(e)}function ur(e){return new oq(e)}function un(e){return new oj(e)}function ua(e){return new oK(e)}function us(e){return new oZ(e)}function ui(e){return new oJ(e)}function uo(e){return new oQ(e)}function ul(e){return new o1(e)}function uu(e){return new o3(e)}function uh(e){return new o5(e)}function ud(e){return new o7(e)}function up(e){return new o6(e)}function uc(e){return new o8(e)}function uf(e){return new o4(e)}function um(e){return new le(e)}function ug(e){return new lu(e)}function ux(e){return new lh(e)}function uy(e){return new ld(e)}function ub(e){return new ly(e)}function uv(e){return ub(e)}function uw(e){return ub(e)}function uC(e){return new lw(e)}function uI(e){return uC(e)}function uk(e){return uC(e)}function uS(e){return new lk(e)}function uT(e){return uS(e)}function uN(e){return uS(e)}function u$(e){return new lT(e)}function uR(e){return new lR(e)}function uA(e){return new lN(e)}function uE(e){return new lA(e)}function uF(e){return new lx(e)}function uD(e){return new lv(e)}function uO(e){return new lI(e)}function uL(e){return new o_(e)}function uz(e){return new oz(e)}function u_(e){return new oP(e)}function uM(e){return new oM(e)}function uP(e){return new oL(e)}function uB(e){return new oO(e)}function uW(e){return new oH(e)}function uG(e){return new oV(e)}function uU(e){return new oF(e)}function uV(e){return new oB(e)}function uH(e){return new lD(e)}function uq(e){return new lF(e)}function uj(e){return new lt(e)}function uX(e){return new lr(e)}function uK(e){return new ln(e)}function uY(e){return new o0(e)}function uZ(e){return new lO(e)}function uJ(e){return new lP(e)}function uQ(e){return new lU(e)}function u0(e){return new lW(e)}function u1(e){return new lj(e)}lj.className="RandomWidth",nJ.registerClass(lj),e.s(["activation",0,ue,"add",0,uu,"alphaDropout",0,uK,"average",0,uh,"averagePooling1d",0,ub,"averagePooling2d",0,uC,"averagePooling3d",0,uS,"avgPool1d",0,uv,"avgPool2d",0,uI,"avgPool3d",0,uT,"avgPooling1d",0,uw,"avgPooling2d",0,uk,"avgPooling3d",0,uN,"batchNormalization",0,ug,"bidirectional",0,uH,"categoryEncoding",0,u0,"centerCrop",0,uJ,"concatenate",0,ud,"conv1d",0,l1,"conv2d",0,l2,"conv2dTranspose",0,l3,"conv3d",0,l4,"conv3dTranspose",0,l5,"convLstm2d",0,uW,"convLstm2dCell",0,uG,"cropping2D",0,l8,"dense",0,ut,"depthwiseConv2d",0,l9,"dot",0,um,"dropout",0,ur,"elu",0,lK,"embedding",0,ul,"flatten",0,ua,"gaussianDropout",0,uX,"gaussianNoise",0,uj,"globalAveragePooling1d",0,u$,"globalAveragePooling2d",0,uR,"globalMaxPool1d",0,uA,"globalMaxPool2d",0,uE,"globalMaxPooling1d",0,uA,"globalMaxPooling2d",0,uE,"gru",0,uL,"gruCell",0,uz,"inputLayer",0,lX,"layerNormalization",0,ux,"leakyReLU",0,lZ,"lstm",0,u_,"lstmCell",0,uM,"masking",0,uY,"maxPool1d",0,uF,"maxPool2d",0,uD,"maxPooling1d",0,uF,"maxPooling2d",0,uD,"maxPooling3d",0,uO,"maximum",0,up,"minimum",0,uc,"multiply",0,uf,"permute",0,uo,"prelu",0,lJ,"randomWidth",0,u1,"reLU",0,lY,"repeatVector",0,us,"rescaling",0,uZ,"reshape",0,ui,"resizing",0,uQ,"rnn",0,uU,"separableConv2d",0,l6,"simpleRNN",0,uP,"simpleRNNCell",0,uB,"softmax",0,lQ,"spatialDropout1d",0,un,"stackedRNNCells",0,uV,"thresholdedReLU",0,l0,"timeDistributed",0,uq,"upSampling2d",0,l7,"zeroPadding2d",0,uy],96722),e.i(96722),e.s(["Layer",0,sx,"RNN",0,oF,"RNNCell",0,oD,"activation",0,ue,"add",0,uu,"alphaDropout",0,uK,"average",0,uh,"averagePooling1d",0,ub,"averagePooling2d",0,uC,"averagePooling3d",0,uS,"avgPool1d",0,uv,"avgPool2d",0,uI,"avgPool3d",0,uT,"avgPooling1d",0,uw,"avgPooling2d",0,uk,"avgPooling3d",0,uN,"batchNormalization",0,ug,"bidirectional",0,uH,"categoryEncoding",0,u0,"centerCrop",0,uJ,"concatenate",0,ud,"conv1d",0,l1,"conv2d",0,l2,"conv2dTranspose",0,l3,"conv3d",0,l4,"conv3dTranspose",0,l5,"convLstm2d",0,uW,"convLstm2dCell",0,uG,"cropping2D",0,l8,"dense",0,ut,"depthwiseConv2d",0,l9,"dot",0,um,"dropout",0,ur,"elu",0,lK,"embedding",0,ul,"flatten",0,ua,"gaussianDropout",0,uX,"gaussianNoise",0,uj,"globalAveragePooling1d",0,u$,"globalAveragePooling2d",0,uR,"globalMaxPool1d",0,uA,"globalMaxPool2d",0,uE,"globalMaxPooling1d",0,uA,"globalMaxPooling2d",0,uE,"gru",0,uL,"gruCell",0,uz,"input",0,iB,"inputLayer",0,lX,"layerNormalization",0,ux,"leakyReLU",0,lZ,"lstm",0,u_,"lstmCell",0,uM,"masking",0,uY,"maxPool1d",0,uF,"maxPool2d",0,uD,"maxPooling1d",0,uF,"maxPooling2d",0,uD,"maxPooling3d",0,uO,"maximum",0,up,"minimum",0,uc,"multiply",0,uf,"permute",0,uo,"prelu",0,lJ,"randomWidth",0,u1,"reLU",0,lY,"repeatVector",0,us,"rescaling",0,uZ,"reshape",0,ui,"resizing",0,uQ,"rnn",0,uU,"separableConv2d",0,l6,"simpleRNN",0,uP,"simpleRNNCell",0,uB,"softmax",0,lQ,"spatialDropout1d",0,un,"stackedRNNCells",0,uV,"thresholdedReLU",0,l0,"timeDistributed",0,uq,"upSampling2d",0,l7,"zeroPadding2d",0,uy],67327);var u2=e.i(67327);e.s(["MAPE",0,function(e,t){return s0(e,t)},"MSE",0,function(e,t){return sJ(e,t)},"binaryAccuracy",0,function(e,t){return s8(e,t)},"binaryCrossentropy",0,function(e,t){return s3(e,t)},"categoricalAccuracy",0,function(e,t){return s7(e,t)},"categoricalCrossentropy",0,function(e,t){return s1(e,t)},"cosineProximity",0,function(e,t){return s4(e,t)},"mape",0,function(e,t){return s0(e,t)},"meanAbsoluteError",0,function(e,t){return sQ(e,t)},"meanAbsolutePercentageError",0,function(e,t){return s0(e,t)},"meanSquaredError",0,function(e,t){return sJ(e,t)},"mse",0,function(e,t){return sJ(e,t)},"precision",0,function(e,t){return ie(e,t)},"r2Score",0,function(e,t){return(0,n$.tidy)(()=>{let r=e.sub(t).square().sum(),n=e.sub(e.mean()).square().sum();return k.scalar(1).sub(r.div(n))})},"recall",0,function(e,t){return(0,n$.tidy)(()=>{let r=s9(e,t),n=(0,n$.tidy)(()=>y.cast(F.sum(el.logicalAnd(e7.equal(e,1),e7.equal(t,0))),"float32")),a=P.add(r,n);return y.cast(eu.where(e1.greater(a,0),C.div(r,a),0),"float32")})},"sparseCategoricalAccuracy",0,function(e,t){return ir(e,t)}],28034);var u3=e.i(28034);e.s([],11195),e.i(11195),e.s(["modelFromJSON",0,iO],20931);var u4=e.i(20931);e.s(["l1",0,function(e){return i7(e),new oe({l1:null!=e?e.l1:null,l2:0})},"l1l2",0,function(e){return new oe(e)},"l2",0,function(e){return i7(e),new oe({l2:null!=e?e.l2:null,l1:0})}],11471);var u5=e.i(11471);class u6 extends sP{constructor(){super(...arguments),this.model=null}setModel(e){if(!(e instanceof iF))throw Error("model must be a LayersModel, not some other Container");this.model=e}}function u8(e,t){return e<t}function u7(e,t){return e>t}class u9 extends u6{constructor(e){if(super(),null==e&&(e={}),e.restoreBestWeights)throw new nD("restoreBestWeights = True is not implemented in EarlyStopping yet.");this.monitor=e.monitor||"val_loss",this.minDelta=Math.abs(e.minDelta||0),this.patience=e.patience||0,this.verbose=e.verbose||0,this.mode=e.mode||"auto",this.baseline=e.baseline,-1===["auto","min","max"].indexOf(this.mode)&&(console.warn(`EarlyStopping mode '${this.mode}' is invalid. Falling back to mode 'auto'.`),this.mode="auto"),"min"===this.mode?this.monitorFunc=u8:"max"===this.mode||-1!==this.monitor.indexOf("acc")?this.monitorFunc=u7:this.monitorFunc=u8,this.monitorFunc===u8&&(this.minDelta*=-1)}async onTrainBegin(e){this.wait=0,this.stoppedEpoch=0,null!=this.baseline?this.best=this.baseline:this.best=this.monitorFunc===u8?1/0:-1/0}async onEpochEnd(e,t){await s_(t);let r=this.getMonitorValue(t);null!=r&&(this.monitorFunc(r-this.minDelta,this.best)?(this.best=r,this.wait=0):(this.wait++,this.wait>=this.patience&&(this.stoppedEpoch=e,this.model.stopTraining=!0)))}async onTrainEnd(e){this.stoppedEpoch>0&&this.verbose&&console.log(`Epoch ${this.stoppedEpoch}: early stopping.`)}getMonitorValue(e){null==e&&(e={});let t=e[this.monitor];return null==t&&console.warn(`Metric for EarlyStopping ${this.monitor} is not available. Available metrics are: ${Object.keys(e)}`),t}}let he={earlyStopping:function(e){return new u9(e)}};e.s([],48635),e.i(53812);var ht=e.i(6536),hr=e.i(11403),hr=hr;function hn(e,t,r=new Map,n=new Set){if(null==e)return null;if("function"==typeof Blob&&e instanceof Blob)return e.slice();if(n.has(e))throw Error("Circular references are not supported.");if(r.has(e))return r.get(e);let a=t(e);if(a.recurse&&null!==a.value)throw Error("A deep map function may not return both a value and recurse=true.");if(!a.recurse)return r.set(e,a.value),a.value;if(hi(e)){let a=Array.isArray(e)?[]:{};for(let s in n.add(e),e){let i=hn(e[s],t,r,n);a[s]=i}return n.delete(e),e.__proto__&&(a.__proto__=e.__proto__),a}throw Error(`Can't recurse into non-iterable type: ${e}`)}function ha(e){return null===e?null:hi(e[0])?{value:null,recurse:!0}:{value:e,recurse:!1}}async function hs(e,t){let r=new Map;for(let n of(hn(e,t,r),Array.from(r.keys()))){let e=r.get(n);if(nR.util.isPromise(e)){let t=await e;r.set(n,t)}}return hn(e,t,r)}function hi(e){let t=!1;if(nN.env().get("IS_BROWSER"))t=e instanceof TextDecoder;else{let{StringDecoder:r}={};t=e instanceof r}return null!=e&&!ArrayBuffer.isView(e)&&(Array.isArray(e)||"object"==typeof e&&!(e instanceof rm.Tensor)&&!(e instanceof Promise)&&!t)}function ho(e){return e instanceof rm.Tensor?{value:e.clone(),recurse:!1}:hi(e)?{value:null,recurse:!0}:{value:e,recurse:!1}}class hl{constructor(e){if(this.capacity=e,this.begin=0,this.end=0,null==e)throw RangeError("Can't create a ring buffer of unknown capacity.");if(e<1)throw RangeError("Can't create ring buffer of capacity < 1.");this.data=Array(e),this.doubledCapacity=2*e}wrap(e){for(;e<0;)e+=this.doubledCapacity;return e%this.doubledCapacity}get(e){if(e<0)throw RangeError("Can't get item at a negative index.");return this.data[e%this.capacity]}set(e,t){if(e<0)throw RangeError("Can't set item at a negative index.");this.data[e%this.capacity]=t}length(){let e=this.end-this.begin;return e<0&&(e=this.doubledCapacity+e),e}isFull(){return this.length()===this.capacity}isEmpty(){return 0===this.length()}push(e){if(this.isFull())throw RangeError("Ring buffer is full.");this.set(this.end,e),this.end=this.wrap(this.end+1)}pushAll(e){for(let t of e)this.push(t)}pop(){if(this.isEmpty())throw RangeError("Ring buffer is empty.");this.end=this.wrap(this.end-1);let e=this.get(this.end);return this.set(this.end,void 0),e}unshift(e){if(this.isFull())throw RangeError("Ring buffer is full.");this.begin=this.wrap(this.begin-1),this.set(this.begin,e)}shift(){if(this.isEmpty())throw RangeError("Ring buffer is empty.");let e=this.get(this.begin);return this.set(this.begin,void 0),this.begin=this.wrap(this.begin+1),e}shuffleExcise(e){if(this.isEmpty())throw RangeError("Ring buffer is empty.");let t=this.wrap(this.begin+e),r=this.get(t);return this.set(t,this.pop()),r}}class hu extends hl{constructor(){super(hu.INITIAL_CAPACITY)}isFull(){return!1}push(e){super.isFull()&&this.expand(),super.push(e)}unshift(e){super.isFull()&&this.expand(),super.unshift(e)}expand(){let e=2*this.capacity,t=Array(e),r=this.length();for(let e=0;e<r;e++)t[e]=this.get(this.wrap(this.begin+e));this.data=t,this.capacity=e,this.doubledCapacity=2*this.capacity,this.begin=0,this.end=r}}hu.INITIAL_CAPACITY=32;class hh{async toArray(){let e=[],t=await this.next();for(;!t.done;)e.push(t.value),t=await this.next();return e}async toArrayForTest(){let e=this.prefetch(100),t=[],r=await e.next();for(;!r.done;)t.push(r.value),r=await e.next();return t}async resolveFully(){let e=await this.next();for(;!e.done;)e=await this.next()}async resolveWhile(e){let t=await this.next(),r=e(t.value);for(;!t.done&&r;)r=e((t=await this.next()).value)}handleErrors(e){return new hb(this,e)}filter(e){return new hx(this,e)}map(e){return new hy(this,e)}mapAsync(e){return new hv(this,e)}serialMapAsync(e){return new hv(this,e).serial()}flatmap(e){return new hC(this,e)}async forEachAsync(e){return this.map(e).resolveFully()}async serialForEach(e){return this.serialMapAsync(e).resolveWhile(e=>!0===e)}rowMajorBatch(e,t=!0){return new hg(this,e,t)}columnMajorBatch(e,t=!0,r=ha){return this.rowMajorBatch(e,t).map(e=>(function(e,t=ha){return function e(t,r,n=new Set){let a=t[0];if(n.has(a))throw Error("Circular references are not supported.");let s=r(t);if(s.recurse&&null!==s.value)throw Error("A deep zip function may not return both a value and recurse=true.");if(!s.recurse)return s.value;if(hi(a)){let s=Array.isArray(a)?[]:{};for(let i in n.add(a),a){let a=e(t.map(e=>e[i]),r,n);s[i]=a}return n.delete(a),s}throw Error(`Can't recurse into non-iterable type: ${a}`)}(e,t)})(e,r))}concatenate(e,t){return new hI(new hd([this,e]),t)}take(e){return e<0||null==e?this:new hm(this,e)}skip(e){return e<0||null==e?this:new hf(this,e)}prefetch(e){return new hS(this,e)}shuffle(e,t){return new hT(this,e,t)}serial(){return new hc(this)}}class hd extends hh{constructor(e){super(),this.items=e,this.trav=0}summary(){return`Array of ${this.items.length} items`}async next(){if(this.trav>=this.items.length)return{value:null,done:!0};let e=this.items[this.trav];return this.trav++,{value:hn(e,ho),done:!1}}}class hp extends hh{constructor(e){super(),this.nextFn=e}summary(){return"Function call"}async next(){try{return this.nextFn()}catch(e){throw e.message=`Error thrown while iterating through a dataset: ${e.message}`,e}}}class hc extends hh{constructor(e){super(),this.upstream=e,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Serial`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){return this.upstream.next()}}class hf extends hh{constructor(e,t){super(),this.upstream=e,this.maxCount=t,this.count=0,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Skip`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;this.count++<this.maxCount;){let e=await this.upstream.next();if(e.done)return e;n$.dispose(e.value)}return this.upstream.next()}}class hm extends hh{constructor(e,t){super(),this.upstream=e,this.maxCount=t,this.count=0}summary(){return`${this.upstream.summary()} -> Take`}async next(){return this.count++>=this.maxCount?{value:null,done:!0}:this.upstream.next()}}class hg extends hh{constructor(e,t,r=!0){super(),this.upstream=e,this.batchSize=t,this.enableSmallLastBatch=r,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> RowMajorBatch`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){let e=[];for(;e.length<this.batchSize;){let t=await this.upstream.next();if(t.done){if(this.enableSmallLastBatch&&e.length>0)return{value:e,done:!1};return{value:null,done:!0}}e.push(t.value)}return{value:e,done:!1}}}class hx extends hh{constructor(e,t){super(),this.upstream=e,this.predicate=t,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Filter`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;;){let e=await this.upstream.next();if(e.done||this.predicate(e.value))return e;n$.dispose(e.value)}}}class hy extends hh{constructor(e,t){super(),this.upstream=e,this.transform=t}summary(){return`${this.upstream.summary()} -> Map`}async next(){let e=await this.upstream.next();if(e.done)return{value:null,done:!0};let t=hr.getTensorsInContainer(e.value),r=this.transform(e.value),n=hr.getTensorsInContainer(r);for(let e of t)hr.isTensorInList(e,n)||e.dispose();return{value:r,done:!1}}}class hb extends hh{constructor(e,t){super(),this.upstream=e,this.handler=t,this.count=0,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> handleErrors`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;;)try{return await this.upstream.next()}catch(e){if(!this.handler(e))return{value:null,done:!0}}}}class hv extends hh{constructor(e,t){super(),this.upstream=e,this.transform=t}summary(){return`${this.upstream.summary()} -> AsyncMap`}async next(){let e=await this.upstream.next();if(e.done)return{value:null,done:!0};let t=hr.getTensorsInContainer(e.value),r=await this.transform(e.value),n=hr.getTensorsInContainer(r);for(let e of t)hr.isTensorInList(e,n)||e.dispose();return{value:r,done:!1}}}class hw extends hh{constructor(){super(),this.outputQueue=new hu,this.lastRead=Promise.resolve({value:null,done:!1})}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;0===this.outputQueue.length();)if(!await this.pump())return{value:null,done:!0};return{value:this.outputQueue.shift(),done:!1}}}class hC extends hw{constructor(e,t){super(),this.upstream=e,this.transform=t}summary(){return`${this.upstream.summary()} -> Flatmap`}async pump(){let e=await this.upstream.next();if(e.done)return!1;let t=hr.getTensorsInContainer(e.value),r=this.transform(e.value),n=hr.getTensorsInContainer(r);for(let e of(this.outputQueue.pushAll(r),t))hr.isTensorInList(e,n)||e.dispose();return!0}}class hI extends hh{constructor(e,t){super(),this.baseErrorHandler=t,this.lastRead=null,this.iterator=null,this.moreIterators=e}summary(){return"TODO: fill in upstream of chained summaries -> Chained"}async next(){return this.lastRead=this.readFromChain(this.lastRead),this.lastRead}async readFromChain(e){if(await e,null==this.iterator){let e=await this.moreIterators.next();if(e.done)return{value:null,done:!0};this.iterator=e.value,null!=this.baseErrorHandler&&(this.iterator=this.iterator.handleErrors(this.baseErrorHandler))}let t=await this.iterator.next();return t.done?(this.iterator=null,this.readFromChain(e)):t}}(i=p||(p={}))[i.FAIL=0]="FAIL",i[i.SHORTEST=1]="SHORTEST",i[i.LONGEST=2]="LONGEST";class hk extends hh{constructor(e,t=p.FAIL){super(),this.iterators=e,this.mismatchMode=t,this.count=0,this.currentPromise=null}summary(){return"{TODO: fill in upstream of zip summaries} -> Zip"}async nextState(e){await e;let t=0,r=0,n=await hs(this.iterators,function(e){return e instanceof hh?{value:e.next().then(e=>(t++,e.done&&r++,e.value)),recurse:!1}:{value:null,recurse:!0}});if(t===r)return{value:null,done:!0};if(r>0)switch(this.mismatchMode){case p.FAIL:throw Error(`Zipped streams should have the same length. Mismatched at element ${this.count}.`);case p.SHORTEST:return{value:null,done:!0};case p.LONGEST:}return this.count++,{value:n,done:!1}}async next(){return this.currentPromise=this.nextState(this.currentPromise),this.currentPromise}}class hS extends hh{constructor(e,t){super(),this.upstream=e,this.bufferSize=t,this.buffer=new hl(t)}summary(){return`${this.upstream.summary()} -> Prefetch`}refill(){for(;!this.buffer.isFull();){let e=this.upstream.next();this.buffer.push(e)}}next(){return this.refill(),this.buffer.shift()}}class hT extends hS{constructor(e,t,r){super(e,t),this.upstream=e,this.windowSize=t,this.upstreamExhausted=!1,this.random=ht.alea(r||nR.util.now().toString()),this.lastRead=Promise.resolve({value:null,done:!1})}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}randomInt(e){return Math.floor(this.random()*e)}chooseIndex(){return this.randomInt(this.buffer.length())}async serialNext(){for(this.upstreamExhausted||this.refill();!this.buffer.isEmpty();){let e=this.chooseIndex(),t=await this.buffer.shuffleExcise(e);if(!t.done)return this.refill(),t;this.upstreamExhausted=!0}return{value:null,done:!0}}}class hN{constructor(){this.size=null}batch(e,t=!0){let r=this;return nR.util.assert(e>0,()=>`batchSize needs to be positive, but it is
      ${e}`),h$(async()=>(await r.iterator()).columnMajorBatch(e,t,hR),this.size===1/0||null==this.size?this.size:t?Math.ceil(this.size/e):Math.floor(this.size/e))}concatenate(e){let t=this;return h$(async()=>(await t.iterator()).concatenate(await e.iterator()),this.size===1/0||e.size===1/0?1/0:null!=this.size&&null!=e.size?this.size+e.size:null)}filter(e){let t=this;return h$(async()=>(await t.iterator()).filter(t=>n$.tidy(()=>e(t))),this.size===1/0?1/0:null)}async forEachAsync(e){return(await this.iterator()).forEachAsync(e)}map(e){let t=this;return h$(async()=>(await t.iterator()).map(t=>n$.tidy(()=>e(t))),this.size)}mapAsync(e){let t=this;return h$(async()=>(await t.iterator()).mapAsync(e),this.size)}prefetch(e){if(null==e)throw RangeError("`Dataset.prefetch()` requires bufferSize to be specified.");let t=this;return h$(async()=>(await t.iterator()).prefetch(e),this.size)}repeat(e){let t=this;return h$(async()=>new hI(new hp(async()=>({value:await t.iterator(),done:!1})).take(e),void 0),null!=this.size&&e>0?this.size*e:0===e?0:null!=this.size&&(void 0===e||e<0)?1/0:null)}skip(e){let t=this;return h$(async()=>(await t.iterator()).skip(e),null!=this.size&&e>=0&&this.size>=e?this.size-e:null!=this.size&&(this.size<e||void 0===e||e<0)?0:null)}shuffle(e,t,r=!0){if(null==e||e<0)if(null==this.size)throw RangeError("`Dataset.shuffle()` requires bufferSize to be specified.");else throw RangeError(`\`Dataset.shuffle()\` requires bufferSize to be specified.  If your data fits in main memory (for regular JS objects), and/or GPU memory (for \`tf.Tensor\`s), consider setting bufferSize to the dataset size (${this.size} elements)`);let n=this,a=ht.alea(t||nR.util.now().toString());return h$(async()=>{let t=a.int32();return r&&(t+=a.int32()),(await n.iterator()).shuffle(e,t.toString())},this.size)}take(e){let t=this;return h$(async()=>(await t.iterator()).take(e),null!=this.size&&this.size>e?e:null!=this.size&&this.size<=e?this.size:null)}async toArray(){if(this.size===1/0)throw Error("Can not convert infinite data stream to array.");return(await this.iterator()).toArray()}async toArrayForTest(){if(this.size===1/0)throw Error("Can not convert infinite data stream to array.");return(await this.iterator()).toArrayForTest()}}function h$(e,t=null){return new class extends hN{constructor(){super(...arguments),this.size=t}async iterator(){return e()}}}function hR(e){var t,r;if(null===e)return null;return null==(t=e[0])||null===(r=t)||"object"!=typeof r&&"function"!=typeof r||Array.isArray(t)||"object"==typeof t&&t instanceof rm.Tensor||nR.util.isTypedArray(t)?{value:function(e){if(0===e.length)throw Error("Can't make a batch of zero elements.");return e[0]instanceof rm.Tensor?eH.stack(e):lL.tensor(e)}(e),recurse:!1}:{value:null,recurse:!0}}hN.MAX_BUFFER_SIZE=1e4;class hA extends hN{constructor(e){super(),this.input=e}async iterator(){return(await this.input.iterator()).decodeUTF8().split("\n").map(e=>(e.endsWith("\r")&&(e=e.slice(0,-1)),e))}}let hE=Symbol("out"),hF=Symbol("field"),hD=Symbol("quote"),hO=Symbol("quoteafterquote"),hL=Symbol("quoteinquote");class hz extends hN{async columnNames(){return this.columnNamesValidated||await this.setColumnNames(),this.configuredColumnsOnly?Object.keys(this.columnConfigs):this.fullColumnNames}async setColumnNames(){let e=await this.maybeReadHeaderLine();if(this.fullColumnNames||e)this.fullColumnNames&&e&&nR.util.assert(e.length===this.fullColumnNames.length,()=>"The length of provided columnNames ("+this.fullColumnNames.length.toString()+") does not match the length of the header line read from file ("+e.length.toString()+").");else throw Error("Column names must be provided if there is no header line.");this.fullColumnNames||(this.fullColumnNames=e);let t=this.fullColumnNames.reduce((e,t)=>(e[t]=e[t]+1||1,e),{}),r=Object.keys(t).filter(e=>t[e]>1);if(nR.util.assert(0===r.length,()=>"Duplicate column names found: "+r.toString()),this.columnConfigs){for(let e of Object.keys(this.columnConfigs))if(-1===this.fullColumnNames.indexOf(e))throw Error('The key "'+e+'" provided in columnConfigs does not match any of the column names ('+this.fullColumnNames.toString()+").")}this.columnNamesValidated=!0}async maybeReadHeaderLine(){if(!this.hasHeader)return null;{let e=await this.base.iterator(),t=await e.next();if(t.done)throw Error("No data was found for CSV parsing.");let r=t.value;return this.parseRow(r,!1)}}constructor(e,t){super(),this.input=e,this.hasHeader=!0,this.fullColumnNames=null,this.columnNamesValidated=!1,this.columnConfigs=null,this.configuredColumnsOnly=!1,this.delimiter=",",this.delimWhitespace=!1,this.base=new hA(e),t||(t={}),this.hasHeader=!1!==t.hasHeader,this.fullColumnNames=t.columnNames,this.columnConfigs=t.columnConfigs,this.configuredColumnsOnly=t.configuredColumnsOnly,t.delimWhitespace?(nR.util.assert(null==t.delimiter,()=>"Delimiter should not be provided when delimWhitespace is true."),this.delimWhitespace=!0,this.delimiter=" "):this.delimiter=t.delimiter?t.delimiter:","}async iterator(){this.columnNamesValidated||await this.setColumnNames();let e=await this.base.iterator();return this.hasHeader&&(e=e.skip(1)),e.map(e=>this.makeDataElement(e))}makeDataElement(e){let t=this.parseRow(e),r={},n={};for(let a=0;a<this.fullColumnNames.length;a++){let s=this.fullColumnNames[a],i=this.columnConfigs?this.columnConfigs[s]:null;if(!this.configuredColumnsOnly||i){let o=t[a],l=null;if(""===o)if(i&&void 0!==i.default)l=i.default;else if(i&&(i.required||i.isLabel))throw Error(`Required column ${s} is empty in this line: ${e}`);else l=void 0;else{let e=Number(o);if(isNaN(e))l=i&&"bool"===i.dtype?this.getBoolean(o):o;else if(i&&i.dtype)switch(i.dtype){case"float32":default:l=e;break;case"int32":l=Math.floor(e);break;case"bool":l=this.getBoolean(o)}else l=e}i&&i.isLabel?n[s]=l:r[s]=l}}return 0===Object.keys(n).length?r:{xs:r,ys:n}}getBoolean(e){return+("1"===e||"true"===e.toLowerCase())}parseRow(e,t=!0){let r=[],n=0,a=e.length,s=hE;for(let t=0;t<a;t++)switch(s){case hE:switch(e.charAt(t)){case'"':n=t+1,s=hD;break;case this.delimiter:if(n=t+1," "===this.delimiter&&this.delimWhitespace)break;r.push(""),s=hE;break;default:s=hF,n=t}break;case hF:e.charAt(t)===this.delimiter&&(r.push(e.substring(n,t)),s=hE,n=t+1);break;case hD:'"'===e.charAt(t)&&(s=hO);break;case hO:switch(e.charAt(t)){case this.delimiter:r.push(e.substring(n,t-1)),s=hE,n=t+1;break;case'"':s=hD;break;default:s=hL}break;case hL:'"'===e.charAt(t)&&(s=hD)}if(s===hO?r.push(e.substring(n,a-1)):r.push(e.substring(n)),t&&r.length!==this.fullColumnNames.length)throw Error(`Invalid row in csv file. Should have ${this.fullColumnNames.length} elements in a row, but got ${r}`);return r}}class h_ extends hh{constructor(e){super(),this.microphoneConfig=e,this.isClosed=!1,this.fftSize=e.fftSize||1024;const t=Math.log2(this.fftSize);if(this.fftSize<0||t<4||t>14||!Number.isInteger(t))throw Error(`Invalid fftSize: it must be a power of 2 between 2 to 4 and 2 to 14, but got ${this.fftSize}`);if(this.numFrames=e.numFramesPerSpectrogram||43,this.sampleRateHz=e.sampleRateHz,this.columnTruncateLength=e.columnTruncateLength||this.fftSize,this.audioTrackConstraints=e.audioTrackConstraints,this.smoothingTimeConstant=e.smoothingTimeConstant||0,this.includeSpectrogram=!1!==e.includeSpectrogram,this.includeWaveform=!0===e.includeWaveform,!this.includeSpectrogram&&!this.includeWaveform)throw Error("Both includeSpectrogram and includeWaveform are false. At least one type of data should be returned.")}summary(){return"microphone"}static async create(e={}){if(!(0,nN.env)().get("IS_BROWSER"))throw Error("microphone API is only supported in browser environment.");let t=new h_(e);return await t.start(),t}async start(){try{this.stream=await navigator.mediaDevices.getUserMedia({audio:null==this.audioTrackConstraints||this.audioTrackConstraints,video:!1})}catch(e){throw Error(`Error thrown while initializing video stream: ${e.message}`)}if(!this.stream)throw Error("Could not obtain audio from microphone.");let e=window.AudioContext||window.webkitAudioContext;if(this.audioContext=new e,this.sampleRateHz){if(this.audioContext.sampleRate!==this.sampleRateHz)throw Error(`Mismatch in sampling rate: Expected: ${this.sampleRateHz}; Actual: ${this.audioContext.sampleRate}`)}else this.sampleRateHz=this.audioContext.sampleRate;let t=this.audioContext.createMediaStreamSource(this.stream);this.analyser=this.audioContext.createAnalyser(),this.analyser.fftSize=2*this.fftSize,this.analyser.smoothingTimeConstant=this.smoothingTimeConstant,t.connect(this.analyser),this.freqData=new Float32Array(this.fftSize),this.timeData=new Float32Array(this.fftSize)}async next(){let e,t;if(this.isClosed)return{value:null,done:!0};let r=await this.getAudioData();if(this.includeSpectrogram){let t=this.flattenQueue(r.freqDataQueue);e=this.getTensorFromAudioDataArray(t,[this.numFrames,this.columnTruncateLength,1])}if(this.includeWaveform){let e=this.flattenQueue(r.timeDataQueue);t=this.getTensorFromAudioDataArray(e,[this.numFrames*this.fftSize,1])}return{value:{spectrogram:e,waveform:t},done:!1}}async capture(){return(await this.next()).value}async getAudioData(){let e=[],t=[],r=0;return new Promise(n=>{let a=setInterval(()=>{this.includeSpectrogram&&(this.analyser.getFloatFrequencyData(this.freqData),this.freqData[0]===-1/0&&n({freqDataQueue:e,timeDataQueue:t}),e.push(this.freqData.slice(0,this.columnTruncateLength))),this.includeWaveform&&(this.analyser.getFloatTimeDomainData(this.timeData),t.push(this.timeData.slice())),++r===this.numFrames&&(clearInterval(a),n({freqDataQueue:e,timeDataQueue:t}))},this.fftSize/this.sampleRateHz*1e3)})}stop(){!this.isClosed&&(this.isClosed=!0,this.analyser.disconnect(),this.audioContext.close(),null!=this.stream&&this.stream.getTracks().length>0&&this.stream.getTracks()[0].stop())}toArray(){throw Error("Can not convert infinite audio stream to array.")}getSampleRate(){return this.sampleRateHz}flattenQueue(e){let t=e[0].length,r=new Float32Array(e.length*t);return e.forEach((e,n)=>r.set(e,n*t)),r}getTensorFromAudioDataArray(e,t){let r=new Float32Array(nR.util.sizeFromShape(t));return r.set(e,r.length-e.length),(0,lL.tensor)(r,t)}}var hM=e.i(30863),hP=e.i(96796);class hB extends hh{constructor(e,t){if(super(),this.webcamVideoElement=e,this.webcamConfig=t,this.isClosed=!0,this.resize=!1,this.needToResize())if(this.resize=!0,this.cropSize=[this.webcamConfig.resizeHeight,this.webcamConfig.resizeWidth],this.cropBoxInd=(0,aC.tensor1d)([0],"int32"),this.webcamConfig.centerCrop){const e=this.webcamConfig.resizeWidth/this.webcamVideoElement.width,t=this.webcamConfig.resizeHeight/this.webcamVideoElement.height,r=(1-e)/2,n=(1-t)/2;this.cropBox=(0,hP.tensor2d)([n,r,t+n,r+e],[1,4])}else this.cropBox=(0,hP.tensor2d)([0,0,1,1],[1,4])}summary(){return"webcam"}static async create(e,t={}){if(!(0,nN.env)().get("IS_BROWSER"))throw Error("tf.data.webcam is only supported in browser environment.");if(!e){if(e=document.createElement("video"),!t.resizeWidth||!t.resizeHeight)throw Error("Please provide webcam video element, or resizeWidth and resizeHeight to create a hidden video element.");e.width=t.resizeWidth,e.height=t.resizeHeight}let r=new hB(e,t);return await r.start(),r}async start(){this.webcamConfig.facingMode&&nR.util.assert("user"===this.webcamConfig.facingMode||"environment"===this.webcamConfig.facingMode,()=>`Invalid webcam facing mode: ${this.webcamConfig.facingMode}. Please provide 'user' or 'environment'`);try{this.stream=await navigator.mediaDevices.getUserMedia({video:{deviceId:this.webcamConfig.deviceId,facingMode:this.webcamConfig.facingMode?this.webcamConfig.facingMode:"user",width:this.webcamVideoElement.width,height:this.webcamVideoElement.height}})}catch(e){throw e.message=`Error thrown while initializing video stream: ${e.message}`,e}if(!this.stream)throw Error("Could not obtain video from webcam.");try{this.webcamVideoElement.srcObject=this.stream}catch(e){console.log(e),this.webcamVideoElement.src=window.URL.createObjectURL(this.stream)}return this.webcamVideoElement.play(),this.isClosed=!1,new Promise(e=>{this.webcamVideoElement.onloadedmetadata=()=>{e()}})}async next(){let e;if(this.isClosed)return{value:null,done:!0};try{e=hM.browser.fromPixels(this.webcamVideoElement)}catch(e){throw Error(`Error thrown converting video to pixels: ${JSON.stringify(e)}`)}if(!this.resize)return{value:e,done:!1};try{return{value:this.cropAndResizeFrame(e),done:!1}}catch(e){throw Error(`Error thrown cropping the video: ${e.message}`)}finally{e.dispose()}}needToResize(){return!!this.webcamConfig.resizeWidth&&!!this.webcamConfig.resizeHeight&&(this.webcamVideoElement.width!==this.webcamConfig.resizeWidth||this.webcamVideoElement.height!==this.webcamConfig.resizeHeight)}cropAndResizeFrame(e){return(0,n$.tidy)(()=>{let t,r=(0,rl.expandDims)((0,y.cast)(e,"float32"),0),n=(t=al.image.cropAndResize(r,this.cropBox,this.cropBoxInd,this.cropSize,"bilinear")).shape;return(0,E.reshape)(t,n.slice(1))})}async capture(){return(await this.next()).value}stop(){this.stream.getTracks().forEach(e=>e.stop());try{this.webcamVideoElement.srcObject=null}catch(e){console.log(e),this.webcamVideoElement.src=null}this.isClosed=!0}toArray(){throw Error("Can not convert infinite video stream to array.")}}class hW{}var hG=e.i(25918);class hU extends hh{split(e){return new hV(this,e)}}class hV extends hU{constructor(e,t){super(),this.upstream=e,this.impl=new hH(e,t)}summary(){return this.impl.summary()}async next(){return this.impl.next()}}class hH extends hw{constructor(e,t){super(),this.upstream=e,this.separator=t,this.carryover=""}summary(){return`${this.upstream.summary()} -> Split('${this.separator}')`}async pump(){let e=await this.upstream.next();if(e.done)return""!==this.carryover&&(this.outputQueue.push(this.carryover),this.carryover="",!0);let t=e.value.split(this.separator);for(let e of(t[0]=this.carryover+t[0],t.slice(0,-1)))this.outputQueue.push(e);return this.carryover=t[t.length-1],!0}}class hq extends hh{decodeUTF8(){return new hj(this)}}class hj extends hU{constructor(e){super(),this.upstream=e,this.impl=new hX(e)}summary(){return this.impl.summary()}async next(){return this.impl.next()}}class hX extends hw{constructor(e){if(super(),this.upstream=e,(0,nN.env)().get("IS_BROWSER"))this.decoder=new TextDecoder("utf-8");else{const{StringDecoder:e}={};this.decoder=new e("utf8")}}summary(){return`${this.upstream.summary()} -> Utf8`}async pump(){let e,t,r=await this.upstream.next();return!r.done&&(e=r.value,t=(0,nN.env)().get("IS_BROWSER")?this.decoder.decode(e,{stream:!0}):this.decoder.write(hG.Buffer.from(e.buffer)),this.outputQueue.push(t),!0)}}class hK extends hq{constructor(e,t={}){super(),this.file=e,this.options=t,nR.util.assert(e instanceof Uint8Array||!!(0,nN.env)().get("IS_BROWSER")&&(e instanceof File||e instanceof Blob),()=>"FileChunkIterator only supports File, Blob and Uint8Array right now."),this.offset=t.offset||0,this.chunkSize=t.chunkSize||1048576}summary(){return`FileChunks ${this.file}`}async next(){if(this.offset>=(this.file instanceof Uint8Array?this.file.byteLength:this.file.size))return{value:null,done:!0};let e=new Promise((e,t)=>{let r=this.offset+this.chunkSize;if(this.file instanceof Uint8Array)e(new Uint8Array(this.file.slice(this.offset,r)));else{let n=new FileReader;n.onload=r=>{let a=n.result;if(a instanceof ArrayBuffer&&(a=new Uint8Array(a)),!(a instanceof Uint8Array))return t(TypeError("FileReader returned unknown type."));e(a)},n.onabort=e=>t(Error("Aborted")),n.onerror=e=>t(Error(e.type));let a=this.file.slice(this.offset,r);n.readAsArrayBuffer(a)}this.offset=r});return{value:await e,done:!1}}}async function hY(e,t={},r){let n,a;"string"==typeof e?n=e:(n=e.url,a=hZ(e));let s=await (r||nR.util.fetch)(n,a);if(s.ok)return new hK(new Uint8Array(await s.arrayBuffer()),t);throw Error(s.statusText)}let hZ=e=>({method:e.method,headers:e.headers,body:e.body,mode:e.mode,credentials:e.credentials,cache:e.cache,redirect:e.redirect,referrer:e.referrer,integrity:e.integrity});function hJ(e){return"string"==typeof e&&"file://"===e.slice(0,7)}class hQ extends hW{constructor(e,t={}){super(),this.input=e,this.options=t}async iterator(){return hJ(this.input)&&(0,nN.env)().get("IS_NODE")&&(this.input=({}).readFileSync(this.input.slice(7))),new hK(this.input,this.options)}}class h0 extends hW{constructor(e,t={}){super(),this.url=e,this.fileOptions=t}async iterator(){return hJ(this.url)?new hQ(this.url,this.fileOptions).iterator():hY(this.url,this.fileOptions)}}async function h1(e,t){return hB.create(e,t)}async function h2(e){return h_.create(e)}e.s([],57200),e.i(57200);let h3="4.22.0";e.s(["CSVDataset",0,hz,"Dataset",0,hN,"FileDataSource",0,hQ,"TextLineDataset",0,hA,"URLDataSource",0,h0,"array",0,function(e){return h$(async()=>new hd(e),e.length)},"csv",0,function(e,t={}){return new hz(new h0(e),t)},"func",0,function(e){let t=new hp(e);return h$(async()=>t)},"generator",0,function(e){return h$(async()=>{let t=await e();return new hp(()=>t.next())})},"microphone",0,h2,"version_data",0,h3,"webcam",0,h1,"zip",0,function(e){let t;if(!hi(e))throw Error("The argument to zip() must be an object or array.");if(Array.isArray(e))for(let r=0;r<e.length;r++)t=null==t?e[r].size:Math.min(t,e[r].size);else if(e instanceof Object)for(let r in e)t=null==t?e[r].size:Math.min(t,e[r].size);return h$(async()=>(function(e,t=p.FAIL){return new hk(e,t)})(await hs(e,e=>{if(e instanceof hN)return{value:e.iterator(),recurse:!1};if(hi(e))return{value:null,recurse:!0};throw Error("Leaves of the structure passed to zip() must be Datasets, not primitives.")}),p.SHORTEST),t)}],78735);var h4=e.i(78735),tN=tN,h5=e.i(68002),h6=e.i(64371),h8=e.i(89970),h8=h8;function h7(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{null!=e&&nR.util.assert("complex64"!==e.dtype,()=>`${t} does not support complex64 tensors in the CPU backend.`)})}let h9=h8.whereImpl;class de extends h6.KernelBackend{nextDataId(){return de.nextDataId++}constructor(){super(),this.blockSize=48,this.firstUse=!0,this.data=new h6.DataStorage(this,(0,n$.engine)())}write(e,t,r){this.firstUse&&(this.firstUse=!1,(0,nN.env)().get("IS_NODE")&&tN.warn("\n============================\nHi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. \n============================"));let n={id:this.nextDataId()};return this.data.set(n,{values:e,dtype:r,refCount:1}),n}makeTensorInfo(e,t,r){let n;if("string"===t&&null!=r&&r.length>0&&nR.util.isString(r[0])){let a=r.map(e=>nR.util.encodeString(e));n=this.write(a,e,t)}else n=this.write(r,e,t);return{dataId:n,shape:e,dtype:t}}refCount(e){return this.data.has(e)?this.data.get(e).refCount:0}incRef(e){let t=this.data.get(e);t.refCount++}decRef(e){if(this.data.has(e)){let t=this.data.get(e);t.refCount--}}move(e,t,r,n,a){this.data.set(e,{values:t,dtype:n,refCount:a})}numDataIds(){return this.data.numDataIds()}async read(e){return this.readSync(e)}readSync(e){let{dtype:t,complexTensorInfos:r}=this.data.get(e);if("complex64"===t){let e=this.readSync(r.real.dataId),t=this.readSync(r.imag.dataId);return tN.mergeRealAndImagArrays(e,t)}return nR.util.convertBackendValuesAndArrayBuffer(this.data.get(e).values,t)}bufferSync(e){let t=this.readSync(e.dataId);if("string"===e.dtype)try{let r=t.map(e=>nR.util.decodeString(e));return(0,h5.buffer)(e.shape,e.dtype,r)}catch(e){throw Error("Failed to decode encoded string bytes into utf-8")}return(0,h5.buffer)(e.shape,e.dtype,t)}makeOutput(e,t,r){return(0,n$.engine)().makeTensorFromTensorInfo(this.makeTensorInfo(t,r,e),this)}disposeData(e,t=!1){if(this.data.has(e)){if(this.data.get(e).refCount--,!t&&this.data.get(e).refCount>0)return!1;let{complexTensorInfos:r}=this.data.get(e);null!=r&&(this.disposeData(r.real.dataId,!0),this.disposeData(r.imag.dataId,!0)),this.data.delete(e)}return!0}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}async time(e){let t=nR.util.now();return e(),{kernelMs:nR.util.now()-t}}memory(){return{unreliable:!0,reasons:["The reported memory is an upper bound. Due to automatic garbage collection, the true allocated memory may be less."]}}where(e){h7([e],"where");let t=this.readSync(e.dataId);return h9(e.shape,t)}dispose(){}floatPrecision(){return 32}epsilon(){return super.epsilon()}}function dt(e){let t=new Float32Array(e.length);for(let r=0;r<e.length;++r)t[r]=Math.abs(e[r]);return t}de.nextDataId=0,e.s([],57919),e.i(57919);let dr={kernelName:x.Abs,backendName:"cpu",kernelFunc:e=>{let{x:t}=e.inputs,r=e.backend;h7(t,"abs");let n=new Float32Array(nR.util.sizeFromShape(t.shape));return n=dt(r.data.get(t.dataId).values),r.makeOutput(n,t.shape,t.dtype)}};var tN=tN;function dn(e){return(t,r,n,a,s)=>{let i=tN.assertAndGetBroadcastShape(t,r),o=i.length,l=nR.util.computeStrides(i),u=nR.util.sizeFromShape(i),h=nR.util.getTypedArrayFromDType(s,u),d=t.length,p=r.length,c=nR.util.computeStrides(t),f=nR.util.computeStrides(r),m=tN.getBroadcastDims(t,i),g=tN.getBroadcastDims(r,i);if(m.length+g.length===0)for(let t=0;t<h.length;++t)h[t]=e(n[t%n.length],a[t%a.length]);else for(let t=0;t<h.length;++t){let r=nR.util.indexToLoc(t,o,l),s=r.slice(-d);m.forEach(e=>s[e]=0);let i=nR.util.locToIndex(s,d,c),u=r.slice(-p);g.forEach(e=>u[e]=0);let x=nR.util.locToIndex(u,p,f);h[t]=e(n[i],a[x])}return[h,i]}}var tN=tN;function da(e){let{inputs:t,backend:r}=e,{real:n,imag:a}=t,s=r.data.get(n.dataId).values,i=r.data.get(a.dataId).values,o=r.makeTensorInfo(n.shape,"complex64");return r.data.get(o.dataId).complexTensorInfos={real:r.makeTensorInfo(n.shape,"float32",s),imag:r.makeTensorInfo(a.shape,"float32",i)},o}let ds={kernelName:x.Complex,backendName:"cpu",kernelFunc:da};function di(e,t,r="float32"){if("complex64"===r)return da({inputs:{real:di(e,t,"float32"),imag:di(e,t,"float32")},backend:e});let n=nR.util.makeZerosTypedArray(nR.util.sizeFromShape(t),r);return e.makeTensorInfo(t,r,n)}function dl(e){let{inputs:t,backend:r}=e,{x:n}=t;return r.incRef(n.dataId),{dataId:n.dataId,shape:n.shape,dtype:n.dtype}}let du={kernelName:x.Identity,backendName:"cpu",kernelFunc:dl};function dh(e){let{inputs:t,backend:r}=e,{input:n}=t,a=r.data.get(n.dataId).complexTensorInfos.real,s=r.data.get(a.dataId).values;return r.makeTensorInfo(a.shape,a.dtype,s)}let dd={kernelName:x.Real,backendName:"cpu",kernelFunc:dh};function dp(e,t,r,n){if("int32"===n)return[t,"int32",Int32Array.from(e)];if("bool"===n){let n=nR.util.toTypedArray([0],r),[a,s]=dn((e,t)=>+(e!==t))(t,[],e,n,"bool");return[s,"bool",a]}throw Error(`Error in Cast: failed to cast ${r} to ${n}`)}function dc(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{dtype:s}=n;if("complex64"===s){if("complex64"===a.dtype)return dl({inputs:{x:a},backend:r});let e=di(r,a.shape,a.dtype),t=dc({inputs:{x:a},backend:r,attrs:{dtype:"float32"}}),n=da({inputs:{real:t,imag:e},backend:r});return r.disposeIntermediateTensorInfo(e),r.disposeIntermediateTensorInfo(t),n}if("complex64"===a.dtype){let e=dh({inputs:{input:a},backend:r}),t=dc({inputs:{x:e},backend:r,attrs:{dtype:s}});return r.disposeIntermediateTensorInfo(e),t}if(!nR.util.hasEncodingLoss(a.dtype,s)){let e=dl({inputs:{x:a},backend:r});return{dataId:e.dataId,shape:e.shape,dtype:s}}let[i,o,l]=dp(r.data.get(a.dataId).values,a.shape,a.dtype,s);return r.makeTensorInfo(i,o,l)}let df={kernelName:x.Cast,backendName:"cpu",kernelFunc:dc};function dm(e,t,r,n){return null==r?({inputs:r,backend:a})=>{let{a:s,b:i}=r;h7([s,i],e);let o=a.data.get(s.dataId).values,l=a.data.get(i.dataId).values,u="string"===s.dtype?tN.fromUint8ToStringArray(o):o,h="string"===s.dtype?tN.fromUint8ToStringArray(l):l,d=n||s.dtype,[p,c]=t(s.shape,i.shape,u,h,d);return a.makeTensorInfo(c,d,p)}:({inputs:e,backend:a})=>{let{a:s,b:i}=e;if("complex64"===s.dtype||"complex64"===i.dtype){let e=dc({inputs:{x:s},backend:a,attrs:{dtype:"complex64"}}),t=a.data.get(e.dataId),n=t.complexTensorInfos.real,o=t.complexTensorInfos.imag,l=a.data.get(n.dataId).values,u=a.data.get(o.dataId).values,h=dc({inputs:{x:i},backend:a,attrs:{dtype:"complex64"}}),d=a.data.get(h.dataId),p=d.complexTensorInfos.real,c=d.complexTensorInfos.imag,f=a.data.get(p.dataId).values,m=a.data.get(c.dataId).values,[g,x,y]=r(s.shape,i.shape,l,u,f,m),b=a.makeTensorInfo(y,"float32",g),v=a.makeTensorInfo(y,"float32",x),w=da({inputs:{real:b,imag:v},backend:a});return a.disposeIntermediateTensorInfo(e),a.disposeIntermediateTensorInfo(h),a.disposeIntermediateTensorInfo(b),a.disposeIntermediateTensorInfo(v),w}{let e=a.data.get(s.dataId).values,r=a.data.get(i.dataId).values,o=n||s.dtype,[l,u]=t(s.shape,i.shape,e,r,o);return a.makeTensorInfo(u,o,l)}}}function dg(e){return(t,r,n,a,s,i)=>{let o=tN.assertAndGetBroadcastShape(t,r),l=nR.util.sizeFromShape(o),u=o.length,h=nR.util.computeStrides(o),d=nR.util.getTypedArrayFromDType("float32",l),p=nR.util.getTypedArrayFromDType("float32",l),c=tN.getBroadcastDims(t,o),f=tN.getBroadcastDims(r,o),m=tN.mergeRealAndImagArrays(n,a),g=tN.mergeRealAndImagArrays(s,i),x=t.length,y=nR.util.computeStrides(t),b=r.length,v=nR.util.computeStrides(r);if(c.length+f.length===0)for(let t=0;t<d.length;t++){let r=t%m.length,n=t%g.length,a=e(m[2*r],m[2*r+1],g[2*n],g[2*n+1]);d[t]=a.real,p[t]=a.imag}else for(let t=0;t<d.length;t++){let r=nR.util.indexToLoc(t,u,h),n=r.slice(-x);c.forEach(e=>n[e]=0);let a=nR.util.locToIndex(n,x,y),s=r.slice(-b);f.forEach(e=>s[e]=0);let i=nR.util.locToIndex(s,b,v),o=e(m[2*a],m[2*a+1],g[2*i],g[2*i+1]);d[t]=o.real,p[t]=o.imag}return[d,p,o]}}let dx=dn((e,t)=>e+t),dy=dg((e,t,r,n)=>({real:e+r,imag:t+n})),db=dm(x.Add,dx,dy),dv={kernelName:x.Add,backendName:"cpu",kernelFunc:db};function dw(e,t,r,n,a){let s=nR.util.sizeFromShape(n),i=nR.util.makeZerosTypedArray(a,r);for(let r=0;r<e.length;r++){let n=e[r];if(n<0)throw Error("Input x must be non-negative!");n>=a||(s>0?i[n]+=t[r]:i[n]+=1)}return i}function dC(e,t,r,n=!1){let a=e.shape[0],s=e.shape[1],i=(0,h5.buffer)([a,r],t.dtype);for(let o=0;o<a;o++)for(let a=0;a<s;a++){let s=e.get(o,a);if(s<0)throw Error("Input x must be non-negative!");s>=r||(n?i.set(1,o,s):t.size>0?i.set(i.get(o,s)+t.get(o,a),o,s):i.set(i.get(o,s)+1,o,s))}return i}let dI=dn((e,t)=>e&t),dk=dm(x.BitwiseAnd,dI),dS={kernelName:x.BitwiseAnd,backendName:"cpu",kernelFunc:dk};function dT(e){return(t,r,n)=>{let a=nR.util.getArrayFromDType(r,t.length);for(let r=0;r<t.length;++r)a[r]=e(t[r],n);return a}}var tN=tN;function dN(e,t,r){return d$(e,dT(t),r)}function d$(e,t,r){return({inputs:n,attrs:a,backend:s})=>{let i,{x:o}=n;h7(o,e);let l=s.data.get(o.dataId).values;if("string"===o.dtype){if(!Array.isArray(l))throw Error("String tensor's value was not an instance of Array");i=tN.fromUint8ToStringArray(l)}else i=l;let u=r||o.dtype,h=t(i,u,a);return s.makeTensorInfo(o.shape,u,h)}}let dR=dT(e=>Math.ceil(e)),dA=d$(x.Ceil,dR),dE={kernelName:x.Ceil,backendName:"cpu",kernelFunc:dA};var tN=tN;function dF(e,t,r,n){let a=nR.util.getArrayFromDType(r,nR.util.sizeFromShape(t));if(n&&"string"!==r){let t=0;e.forEach(e=>{let r=nR.util.sizeFromShape(e.shape);a.set(e.vals,t),t+=r})}else{let n=0;e.forEach(e=>{let s="string"===r?tN.fromUint8ToStringArray(e.vals):e.vals,i=0;for(let r=0;r<e.shape[0];++r){let o=r*t[1]+n;for(let t=0;t<e.shape[1];++t)a[o+t]=s[i++]}n+=e.shape[1]})}return a}let dD=dn((e,t)=>+(e===t)),dO=dm(x.Equal,dD,null,"bool"),dL={kernelName:x.Equal,backendName:"cpu",kernelFunc:dO},dz=dT(e=>Math.exp(e)),d_=d$(x.Exp,dz,"float32"),dM={kernelName:x.Exp,backendName:"cpu",kernelFunc:d_},dP=dT(e=>Math.expm1(e)),dB=d$(x.Expm1,dP),dW={kernelName:x.Expm1,backendName:"cpu",kernelFunc:dB},dG=dT(e=>Math.floor(e)),dU=d$(x.Floor,dG),dV={kernelName:x.Floor,backendName:"cpu",kernelFunc:dU},dH=dn((e,t)=>Math.floor(e/t)),dq=dm(x.FloorDiv,dH,null,"int32"),dj={kernelName:x.FloorDiv,backendName:"cpu",kernelFunc:dq};function dX(e,t,r,n,a,s,i,o,l){let u=(0,h5.buffer)([n,s],r);for(let r=0;r<n;r++){let n=[],h=0;for(let t=0;t<a;t++){let s=e[r*a+t];h+=s*i[t],n.push(s)}if(h<0||h>=l/s)throw Error(`Invalid indices: ${n} does not index into ${o}`);for(let e=0;e<s;e++)u.values[r*s+e]=t.get(...t.indexToLoc(h*s+e))}return u}function dK(e,t,r){let n=(0,h5.buffer)(r,e.dtype);for(let r=0;r<n.size;++r){let a=n.indexToLoc(r).slice(),s=a[0],i=a[2],o=t.locToIndex([s,i]);a[2]=t.values[o];let l=e.locToIndex(a);0<=l&&l<e.values.length&&(n.values[r]=e.values[l])}return n}let dY=dn((e,t)=>+(e>t)),dZ=dm(x.Greater,dY,null,"bool"),dJ={kernelName:x.Greater,backendName:"cpu",kernelFunc:dZ},dQ=dn((e,t)=>+(e>=t)),d0=dm(x.GreaterEqual,dQ,null,"bool"),d1={kernelName:x.GreaterEqual,backendName:"cpu",kernelFunc:d0},d2=dn((e,t)=>+(e<t)),d3=dm(x.Less,d2,null,"bool"),d4={kernelName:x.Less,backendName:"cpu",kernelFunc:d3},d5=dn((e,t)=>+(e<=t)),d6=dm(x.LessEqual,d5,null,"bool"),d8={kernelName:x.LessEqual,backendName:"cpu",kernelFunc:d6};function d7(e,t,r){let n=(t-e)/(r-1),a=nR.util.makeZerosTypedArray(r,"float32");a[0]=e;for(let e=1;e<a.length;e++)a[e]=a[e-1]+n;return a}let d9=dT(e=>Math.log(e)),pe=d$(x.Log,d9),pt={kernelName:x.Log,backendName:"cpu",kernelFunc:pe};function pr(e,t,r,n){let a=nR.util.getTypedArrayFromDType(n,nR.util.sizeFromShape(r));for(let r=0;r<a.length;++r){let n=r*t,s=e[n];for(let r=0;r<t;++r){let t=e[n+r];(Number.isNaN(t)||t>s)&&(s=t)}a[r]=s}return a}let pn=dn((e,t)=>Math.max(e,t)),pa=dm(x.Maximum,pn),ps={kernelName:x.Maximum,backendName:"cpu",kernelFunc:pa},pi=dn((e,t)=>Math.min(e,t)),po=dm(x.Minimum,pi),pl={kernelName:x.Minimum,backendName:"cpu",kernelFunc:po},pu=dn((e,t)=>e*t),ph=dg((e,t,r,n)=>({real:e*r-t*n,imag:e*n+t*r})),pd=dm(x.Multiply,pu,ph),pp={kernelName:x.Multiply,backendName:"cpu",kernelFunc:pd};function pc(e,t,r){return pu([],t,nR.util.createScalarValue(-1,r),e,r)}let pf={kernelName:x.Neg,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{x:n}=t;h7(n,"neg");let[a,s]=pc(r.data.get(n.dataId).values,n.shape,n.dtype);return r.makeTensorInfo(s,n.dtype,a)}},pm=dn((e,t)=>+(e!==t)),pg=dm(x.NotEqual,pm,null,"bool"),px={kernelName:x.NotEqual,backendName:"cpu",kernelFunc:pg};var tN=tN,py=e.i(71750);function pb(e,t,r,n,a){let s=t.length,i=nR.util.sizeFromShape(t),o=nR.util.computeStrides(t),l=nR.util.computeStrides(a),u=nR.util.getTypedArrayFromDType(r,nR.util.sizeFromShape(a));for(let t=0;t<i;++t){let r=nR.util.indexToLoc(t,s,o),a=Array(r.length);for(let e=0;e<a.length;e++)a[e]=r[n[e]];u[nR.util.locToIndex(a,s,l)]=e[t]}return u}function pv(e){let{inputs:t,attrs:r,backend:n}=e,{x:a}=t,{perm:s}=r;h7(a,"transpose");let i=Array(a.shape.length);for(let e=0;e<i.length;e++)i[e]=a.shape[s[e]];let o=pb(n.data.get(a.dataId).values,a.shape,a.dtype,s,i);return{dataId:n.write(o,i,a.dtype),shape:i,dtype:a.dtype}}let pw={kernelName:x.Transpose,backendName:"cpu",kernelFunc:pv};function pC(e,t,r,n){let[a,s]=tN.computeOutAndReduceShapes(e,n),i=(0,py.upcastType)(t,"int32"),o=nR.util.makeZerosTypedArray(nR.util.sizeFromShape(a),i),l=nR.util.sizeFromShape(s);for(let e=0;e<o.length;++e){let t=e*l,n=1;for(let e=0;e<l;++e)n*=r[t+e];o[e]=n}return{outVals:o,outShape:a,outDtype:i}}let pI={kernelName:x.Prod,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,keepDims:i}=n;h7(a,"prod");let o=a.shape.length,l=nR.util.parseAxisParam(s,a.shape),u=tN.getAxesPermutation(l,o),h=l,d=a,p=[];null!=u&&(p.push(d=pv({inputs:{x:a},backend:r,attrs:{perm:u}})),h=tN.getInnerMostAxes(h.length,o));let c=r.data.get(d.dataId).values,{outVals:f,outShape:m,outDtype:g}=pC(d.shape,d.dtype,c,h),x=m;return i&&(x=tN.expandShapeToKeepDim(m,l)),p.forEach(e=>r.disposeIntermediateTensorInfo(e)),r.makeTensorInfo(x,g,f)}};function pk(e,t){let r=e.slice(0,t);for(;r.length<t;)r.push(1);for(let n=t;n<e.length;n++)r[t-1]*=e[n];return r}function pS(e,t,r,n,a,s,i,o){let l,u,h,d;if(0===e.length)throw Error("paramsNestedSplits must be non empty");if(0===t[0].length)throw Error("Split tensors must not be scalars");let p=t[0][0]-1;if(s.forEach((e,t)=>{if(e<0||e>=p){let r=nR.util.indexToLoc(t,i.length,nR.util.computeStrides(i)).join(",");throw Error(`indices[${r}] = ${e} is not in [0, ${p})`)}}),0===n.length)throw Error("params.rank must be nonzero");let{outSplits:c,valueSlices:f,numValues:m}=function(e,t,r,n){let a=[],s=0,i=Array(t.length-1+r.length).fill(null).map(()=>[0]);for(let e=0;e<r.length;++e){let t=r[e],a=e===r.length-1?n:r[e+1].length;if(0===t.length)throw Error("Ragged splits may not be empty");if(t[0]<0)throw Error("Ragged splits must be non-negative");if(t[t.length-1]>a)throw Error("Ragged splits must not point past values");for(let e=1;e<t.length;++e)if(t[e-1]>t[e])throw Error("Ragged splits must be sorted in ascending order")}let o=1;for(let e=0;e<t.length-1;++e){o*=t[e];let r=t[e+1];for(let t=1;t<o+1;++t)i[e].push(t*r)}for(let n=0;n<e.length;++n){let o=e[n],l=e[n]+1;for(let e=0;e<r.length;++e){let n=r[e],a=e+t.length-1;if(a>=0){let e=i[a],t=e[e.length-1]-n[o];for(let e=o;e<l;++e)i[a].push(n[e+1]+t)}o=n[o],l=n[l]}l!==o&&(a.push([o,l]),s+=l-o)}return{outSplits:i,valueSlices:a,numValues:s}}(s,i,e,n[0]),g=function(e){let t=[];for(let r=0;r<e.length;++r){let n=e[r].length,a=nR.util.getArrayFromDType("int32",n);t.push(a),e[r].forEach((e,t)=>a[t]=e)}return t}(c),x=((l=n.slice())[0]=m,u=nR.util.getArrayFromDType(a,nR.util.sizeFromShape(l)),d=0===(h=r.length)?0:h/n[0],!function(e,t,r,n,a,s){let i=pk(t,2)[1],o=pk(s,2)[1],l=0;for(let t of r)for(let r=t[0];r<t[1];++r){for(let t=0;t<n;++t)a[l*o+t]=e[r*i+t];++l}}(r,n,f,d,u,l),[u,l]);return[g,x[0],x[1]]}function pT(e,t,r,n,a,s,i){if(t.length>1)throw Error("starts must be a scalar or vector");if(a.length>1)throw Error("limits must be a scalar or vector");if(i.length>1)throw Error("deltas must be a scalar or vector");let o=0===t.length,l=0===a.length,u=0===i.length,h=[];o||h.push(t[0]),l||h.push(a[0]),u||h.push(i[0]);for(let e=1;e<h.length;++e)if(h[e]!==h[e-1])throw Error("starts, limits, and deltas must have the same shape");let d=0===h.length?1:h[0],p=nR.util.getArrayFromDType("int32",d+1);p[0]=0;for(let t=0;t<d;++t){let r,a=o?e[0]:e[t],i=l?n[0]:n[t],h=u?s[0]:s[t];if(0===h)throw Error("Requires delta != 0");if(h>0&&i<a||h<0&&i>a)r=0;else if((r=Math.ceil(Math.abs((i-a)/h)))>0x7fffffff)throw Error("Requires ((limit - start) / delta) <= 2147483647");p[t+1]=p[t]+r}let c=p[d],f=nR.util.getArrayFromDType(r,c),m=0;for(let t=0;t<d;++t){let r=p[t+1]-p[t],n=o?e[0]:e[t],a=u?s[0]:s[t];for(let e=0;e<r;++e)f[m++]=n,n+=a}return[p,f]}var tN=tN,pN=tN.RowPartitionType;class p${constructor(e,t,r,n,a,s,i,o,l,u){this.shape=e,this.shapeShape=t,this.values=r,this.valuesShape=n,this.valuesDType=a,this.defaultValue=s,this.defaultValueShape=i,this.rowPartitionValues=o,this.rowPartitionValuesShapes=l,this.rowPartitionTypes=tN.getRowPartitionTypesHelper(u),this.raggedRank=tN.getRaggedRank(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===pN.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===pN.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(e){let t=this.getRowPartitionTensor(e-1);switch(this.getRowPartitionTypeByDimension(e-1)){case pN.VALUE_ROWIDS:return p$.getMaxWidthValueRowID(t);case pN.ROW_SPLITS:return p$.getMaxWidthRowSplit(t);default:throw Error(`Cannot handle partition type ${pN[this.getRowPartitionTypeByDimension(e-1)]}`)}}static getMaxWidthRowSplit(e){let t=e.length;if(0===t||1===t)return 0;let r=0;for(let n=0;n<t-1;++n){let t=e[n+1]-e[n];t>r&&(r=t)}return r}static getMaxWidthValueRowID(e){let t=e.length;if(0===t)return 0;let r=0,n=e[0],a=0;for(let s=1;s<t;++s){let t=e[s];t!==n&&(n=t,a=Math.max(s-r,a),r=s)}return Math.max(t-r,a)}tensorShapeFromTensor(e,t,r=!0){if(0===t.length){if(-1===e[0])return[];throw Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return pA(e,r)}calculateOutputSize(e){let t=this.valuesShape,r=this.defaultValueShape;tN.validateDefaultValueShape(r,t);let n=this.tensorShapeFromTensor(this.shape,this.shapeShape),a=tN.combineRaggedTensorToTensorShapes(this.raggedRank,n,t);a[0]<0&&(a[0]=e);for(let e=1;e<=this.raggedRank;++e)a[e]<0&&(a[e]=this.getMaxWidth(e));return a}calculateFirstParentOutputIndex(e,t,r){let n=Math.min(e,r),a=[],s=0;for(let e=0;e<n;++e,s+=t)a.push(s);for(let t=n;t<e;++t)a.push(-1);return nR.util.assert(a.length===e,()=>"Final length of result must be equal to firstDimension."),a}calculateOutputIndexRowSplit(e,t,r,n){let a=e.length,s=[];for(let i=0;i<a-1;++i){let a=e[i+1]-e[i],o=Math.min(n,a),l=t[i];-1===l&&(o=0);for(let e=0;e<o;++e)s.push(l),l+=r;for(let e=0;e<a-o;++e)s.push(-1)}if(a>0&&s.length!==e[a-1])throw Error("Invalid row split size.");return s}calculateOutputIndexValueRowID(e,t,r,n){let a=e.length,s=[];if(0===a)return[];let i=0,o=e[0];if(o>=t.length)throw Error(`Got currentValueRowId=${o}, which is not less than ${t.length}`);let l=t[o];s.push(l);for(let u=1;u<a;++u){let a=e[u];if(a===o)l>=0&&(++i<n?l+=r:l=-1);else{if(i=0,o=a,a>=t.length)throw Error(`Got nextValueRowId=${a} which is not less than ${t.length}`);l=t[a]}s.push(l)}if(s.length!==e.length)throw Error("Invalid row ids.");return s}calculateOutputIndex(e,t,r,n){let a=this.getRowPartitionTensor(e),s=this.getRowPartitionTypeByDimension(e);switch(s){case pN.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(a,t,r,n);case pN.ROW_SPLITS:if(a.length-1>t.length)throw Error(`Row partition size is greater than output size: ${a.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(a,t,r,n);default:throw Error(`Unsupported partition type: ${pN[s]}`)}}getFirstDimensionSize(){let e=this.rowPartitionValues[0];if(0===this.rowPartitionTypes.length)throw Error("No row_partition_types given.");let t=this.rowPartitionTypes[0];switch(t){case pN.FIRST_DIM_SIZE:return e[0];case pN.VALUE_ROWIDS:throw Error("Cannot handle VALUE_ROWIDS in first dimension.");case pN.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw Error(`Cannot handle type ${pN[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw Error("Invalid first partition input. Tensor requires at least one element.");let e=this.getFirstDimensionSize(),t=this.calculateOutputSize(e),r=Array(this.raggedRank+1);r[r.length-1]=1;for(let e=r.length-2;e>=0;--e)r[e]=r[e+1]*t[e+1];let n=pA(t,!1),a=nR.util.getArrayFromDType(this.valuesDType,nR.util.sizeFromShape(n));if(r[0]*t[0]>0){let s=this.calculateFirstParentOutputIndex(e,r[0],t[0]);for(let e=1;e<=this.raggedRank;++e)s=this.calculateOutputIndex(e-1,s,r[e],t[e]);this.setOutput(this.raggedRank,s,a,n)}return[n,a]}setOutput(e,t,r,n){if(0===r.length)return;let a=this.values,s=n.slice();s=s.slice(e+1);let i=nR.util.sizeFromShape(s),o=t.length,l=this.defaultValue;if(l.length!==i&&1!==l.length){let e=this.defaultValueShape;(0,n$.tidy)(()=>{let t=(0,E.reshape)(l,e);l=(0,rR.broadcastTo)(t,s).dataSync()})}let u=0,h=0,d=0;for(let e=0;e<=o;++e){let n=e<o?t[e]:-1;if(n===d){++d;continue}if(h<d){let e=a.subarray(u*i);pR(r.subarray(h*i),e,(d-h)*i)}if(e>=o&&(n=Math.floor(r.length/i)),n>d)if(1===this.defaultValue.length)r.subarray(d*i,n*i).fill(this.defaultValue[0]),d=n;else for(;n>d;)pR(r.slice(d*i),l,i),++d;n<0?(u=e+1,h=d):(u=e,d=(h=d)+1)}}}function pR(e,t,r){for(let n=0;n<r;n++)e[n]=t[n]}function pA(e,t){let r=[];for(let n of e){if(n<0){if(!t)throw Error(`Dimension ${n} must be >= 0`);if(n<-1)throw Error(`Dimension ${n} must be >= -1`);n=-1}r.push(n)}return r}function pE(e,t,r,n,a,s,i,o,l,u){return new p$(e,t,r,n,a,s,i,o,l,u).compute()}function pF(e,t,r,n){let a=e===t,s=e<t&&r<0,i=t<e&&r>1;if(a||s||i)return nR.util.makeZerosTypedArray(0,n);let o=Math.abs(Math.ceil((t-e)/r)),l=nR.util.makeZerosTypedArray(o,n);t<e&&1===r&&(r=-1),l[0]=e;for(let e=1;e<l.length;e++)l[e]=l[e-1]+r;return l}let pD=dT(e=>1/Math.sqrt(e)),pO=d$(x.Rsqrt,pD),pL={kernelName:x.Rsqrt,backendName:"cpu",kernelFunc:pO};function pz(e,t,r,n,a,s,i,o,l,u){let h=e.values,d=t.values;if(0===n)return(0,h5.buffer)(r,t.dtype);let p=l instanceof rm.TensorBuffer?l:(0,h5.buffer)([n/a,a],t.dtype);"string"==typeof l||"number"==typeof l?p.values.fill(l):"boolean"==typeof l&&p.values.fill(+l);for(let e=0;e<s;e++){let s=[],l=0;for(let t=0;t<i;t++){let r=h[e*i+t];s.push(r),l+=r*o[t]}if(l<0||l>=n/a)throw Error(`Invalid indices: ${s} does not index into ${r}`);for(let r=0;r<a;r++)u?p.values[l*a+r]+=d[e*a+r]:p.values[l*a+r]=0===t.rank?d[0]:d[e*a+r]}return p}let p_=dT(e=>1/(1+Math.exp(-e))),pM=dN(x.Sigmoid,e=>1/(1+Math.exp(-e))),pP={kernelName:x.Sigmoid,backendName:"cpu",kernelFunc:pM};var tN=tN,pB=tJ;function pW(e,t,r,n,a){let s=pB.isSliceContinous(n,t,r),i=nR.util.sizeFromShape(r),o=nR.util.computeStrides(n);if(s){let r=pB.computeFlatOffset(t,o);return"string"===a?e.slice(r,r+i):e.subarray(r,r+i)}let l="string"===a?tN.fromUint8ToStringArray(e):e,u=(0,h5.buffer)(n,a,l),h=(0,h5.buffer)(r,a);for(let e=0;e<h.size;++e){let r=h.indexToLoc(e),n=r.map((e,r)=>e+t[r]);h.set(u.get(...n),...r)}return"string"===a?tN.fromStringArrayToUint8(h.values):h.values}function pG(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{begin:s,size:i}=n;h7(a,"slice");let[o,l]=pB.parseSliceParams(a,s,i);pB.assertParamsValid(a,o,l);let u=pW(r.data.get(a.dataId).values,o,l,a.shape,a.dtype);return r.makeTensorInfo(l,a.dtype,u)}let pU={kernelName:x.Slice,backendName:"cpu",kernelFunc:pG};var tN=tN;function pV(e,t,r,n,a,s,i){let o=t[0],l=s[0],u=Array(l),h=Array(o),d=t[1];if(0===l){if(0!==o)throw Error(tN.getSparseFillEmptyRowsIndicesDenseShapeMismatch(o));return[nR.util.getArrayFromDType(r,0),[0,d],nR.util.getArrayFromDType(a,0),u,h]}let p=!0,c=0,f=Array(l).fill(0);for(let t=0;t<o;++t){let r=e[t*d];if(r<0)throw Error(tN.getSparseFillEmptyRowsNegativeIndexErrorMessage(t,r));if(r>=l)throw Error(tN.getSparseFillEmptyRowsOutOfRangeIndexErrorMessage(t,r,l));++f[r],p=p&&r>=c,c=r}let m=!0;for(let e=0;e<l;++e){let t=0===f[e];u[e]=t,m=m&&!t,f[e]=Math.max(f[e],1),e>0&&(f[e]+=f[e-1])}if(m&&p){for(let e=0;e<o;++e)h[e]=e;return[e,[o,d],n,u,h]}{let t=f[l-1],s=nR.util.getArrayFromDType(r,t*d),p=nR.util.getArrayFromDType(a,t),c=Array(l).fill(0);for(let t=0;t<o;++t){let r=e[t*d],a=c[r],i=(0===r?0:f[r-1])+a;c[r]++;for(let r=0;r<d;++r)s[i*d+r]=e[t*d+r];p[i]=n[t],h[t]=i}for(let e=0;e<l;++e)if(0===c[e]){let t=0===e?0:f[e-1];s[t*d+0]=e;for(let e=1;e<d;++e)s[t*d+e]=0;p[t]=i}return[s,[t,d],p,u,h]}}var tN=tN;function pH(e,t,r,n,a){let s=nR.util.sizeFromShape(n),i=t[0],o=a.length,l=[],u=1,h=-1;for(let e=0;e<o;++e){let t=a[e];if(-1===t){if(-1!==h)throw Error(tN.getSparseReshapeMultipleNegativeOneOutputDimErrorMessage(h,e));h=e,l.push(1)}else{if(t<0)throw Error(tN.getSparseReshapeNegativeOutputDimErrorMessage(e,t));u*=t,l.push(t)}}if(-1!==h){if(u<=0)throw Error(tN.getSparseReshapeEmptyTensorZeroOutputDimErrorMessage());let e=Math.trunc(s/u);if(u*e!==s)throw Error(tN.getSparseReshapeInputOutputMultipleErrorMessage(n,l));l[h]=e}if(nR.util.sizeFromShape(l)!==s)throw Error(tN.getSparseReshapeInputOutputMismatchErrorMessage(n,l));let d=n.length,p=[];if(d>0){p[d-1]=1;for(let e=d-2;e>=0;--e)p[e]=p[e+1]*n[e+1]}let c=[];if(o>0){c[o-1]=1;for(let e=o-2;e>=0;--e)c[e]=c[e+1]*l[e+1]}let f=nR.util.getArrayFromDType(r,i*o);for(let t=0;t<i;++t){let r=0;for(let n=0;n<d;++n)r+=e[t*d+n]*p[n];for(let e=0;e<o;++e)f[t*o+e]=Math.trunc(r/c[e]),r%=c[e]}return[f,[i,o],l]}var tN=tN;function pq(e,t,r,n,a,s=!1,i=0){let o=n.length,l=[t[0],e.length/t[0]],u=l[1],h=o>0?a[o-1]+1:0;if(h<0)throw Error(tN.getSparseSegmentReductionNegativeSegmentIdsErrorMessage());let d=t.slice();d[0]=h;let p=d.reduce((e,t)=>e*t,1),c=nR.util.getArrayFromDType(r,p);if(0===o)return h>0&&c.fill(i),[c,d];if(h<=0)throw Error(tN.getSparseSegmentReductionNegativeSegmentIdsErrorMessage());let f=0,m=1,g=0,x=a[0];for(;;){let t=0;if(m<o){if(x===(t=a[m])){++m;continue}if(x>=t)throw Error(tN.getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage())}if(x<0||x>=h)throw Error(tN.getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage(x,h));x>g&&c.fill(i,g*u,x*u);for(let t=f;t<m;++t){let r=n[t];if(r<0||r>=l[0])throw Error(tN.getSparseSegmentReductionIndicesOutOfRangeErrorMessage(t,n[t],l[0]));for(let t=0;t<u;t++)c[x*u+t]+=e[r*u+t]}if(s)for(let e=0;e<u;e++)c[x*u+e]/=m-f;if(f=m,++m,g=x+1,x=t,m>o)break}return g<h&&c.fill(i,g*u,h*u),[c,d]}let pj=dT(e=>Math.sqrt(e)),pX=dN(x.Sqrt,e=>Math.sqrt(e)),pK={kernelName:x.Sqrt,backendName:"cpu",kernelFunc:pX},pY=dn((e,t)=>{let r=e-t;return r*r}),pZ=dm(x.SquaredDifference,pY),pJ={kernelName:x.SquaredDifference,backendName:"cpu",kernelFunc:pZ},pQ=dT((e,t)=>{let{pattern:r,replaceGlobal:n,rewrite:a}=t;return e.replace(new RegExp(r,n?"g":""),a)}),p0=d$(x.StaticRegexReplace,pQ),p1={kernelName:x.StaticRegexReplace,backendName:"cpu",kernelFunc:p0};function p2(e,t,r,n){let a=(0,h5.buffer)(e,t.dtype);for(let e=0;e<a.size;e++){let s=a.indexToLoc(e),i=Array(s.length);for(let e=0;e<i.length;e++)i[e]=s[e]*r[e]+n[e];a.set(t.get(...i),...s)}return a}class p3{constructor(e,t,r,n,a,s){this.separator=nR.util.encodeString(e),this.nGramWidths=t,this.leftPad=nR.util.encodeString(r),this.rightPad=nR.util.encodeString(n),this.padWidth=a,this.preserveShort=s}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){return Math.max(0,e+2*this.getPadWidth(t)-t+1)}createNGrams(e,t,r,n,a,s){for(let i=0;i<a;++i){let o,l=this.getPadWidth(s),u=Math.max(0,l-i),h=Math.max(0,l-(a-(i+1))),d=s-(u+h),p=t+(u>0?0:i-l);o=0+u*this.leftPad.length;for(let t=0;t<d;++t)o+=e[p+t].length;o+=h*this.rightPad.length,o+=(u+h+d-1)*this.separator.length,r[n+i]=new Uint8Array(o);let c=r[n+i],f=0,m=e=>e.forEach(e=>c[f++]=e);for(let e=0;e<u;++e)m(this.leftPad),m(this.separator);for(let t=0;t<d-1;++t)m(e[p+t]),m(this.separator);if(d>0){m(e[p+d-1]);for(let e=0;e<h;++e)m(this.separator),m(this.rightPad)}else{for(let e=0;e<h-1;++e)m(this.rightPad),m(this.separator);m(this.rightPad)}}}compute(e,t){let r=e.length,n=t.length;if(n>0){let e=t[0];if(0!==e)throw Error(`First split value must be 0, got ${e}`);for(let a=1;a<n;++a){let n=t[a]>=e;if(!(n=n&&t[a]<=r))throw Error(`Invalid split value ${t[a]}, must be in [${e}, ${r}]`);e=t[a]}if(e!==r)throw Error(`Last split value must be data size. Expected ${r}, got ${e}`)}let a=n-1,s=nR.util.getArrayFromDType("int32",n);if(0===r||0===n){let e=Array(r);for(let e=0;e<=a;++e)s[e]=0;return[e,s]}s[0]=0;for(let e=1;e<=a;++e){let r=t[e]-t[e-1],n=0;this.nGramWidths.forEach(e=>{n+=this.getNumNGrams(r,e)}),this.preserveShort&&r>0&&0===n&&(n=1),s[e]=s[e-1]+n}let i=Array(s[a]);for(let r=0;r<a;++r){let n=t[r],a=s[r];if(this.nGramWidths.forEach(s=>{let o=t[r+1]-t[r],l=this.getNumNGrams(o,s);this.createNGrams(e,n,i,a,l,s),a+=l}),this.preserveShort&&a===s[r]){let s=t[r+1]-t[r];if(0===s)continue;let o=s+2*this.padWidth;this.createNGrams(e,n,i,a,1,o)}}return[i,s]}}function p4(e,t,r,n,a,s,i,o){return new p3(r,n,a,s,i,o).compute(e,t)}function p5(e,t,r){let n=e.length,a=[],s=0,i=0,o=Array(n);for(let l=0;l<n;++l){let n=a.length;!function(e,t,r,n){if(!e.length)return;if(0===t.length){for(let t=0;t<e.length;++t)n.push(e.subarray(t,t+1));return}if(1===t.length){let a=t[0],s=e.indexOf(a);for(;-1!==s;){let t=e.subarray(0,s);r&&0===t.length||n.push(t),s=(e=e.subarray(s+1)).indexOf(a)}r&&0===e.length||n.push(e);return}let a=0;for(let s=0;s<e.length+1;s++)if(s===e.length||-1!==t.indexOf(e[s])){let t=e.subarray(a,s);r&&0===t.length||n.push(t),a=s+1}}(e[l],t,r,a);let u=a.length-n;o[l]=u,s+=u,i=Math.max(i,u)}let l=nR.util.getArrayFromDType("int32",2*s),u=Array(s),h=[n,i],d=0;for(let e=0;e<n;++e)for(let t=0;t<o[e];++t)l[2*d]=e,l[2*d+1]=t,u[d]=a[d],++d;return[l,u,h]}function p6(e,t){let r=nR.util.getArrayFromDType("int32",e.length);for(let n=0;n<e.length;++n)r[n]=nR.util.fingerPrint64(e[n]).modulo(t).getLowBitsUnsigned();return r}let p8=dn((e,t)=>e-t),p7=dg((e,t,r,n)=>({real:e-r,imag:t-n})),p9=dm(x.Sub,p8,p7),ce={kernelName:x.Sub,backendName:"cpu",kernelFunc:p9};function ct(e,t){let r=Array(e.rank);for(let n=0;n<r.length;n++)r[n]=e.shape[n]*t[n];let n=(0,h5.buffer)(r,e.dtype);for(let t=0;t<n.values.length;++t){let r=n.indexToLoc(t),a=Array(e.rank);for(let t=0;t<a.length;t++)a[t]=r[t]%e.shape[t];let s=e.locToIndex(a);n.values[t]=e.values[s]}return n}let cr=(e,t)=>{let r=t.value-e.value;return 0===r?e.index-t.index:r};function cn(e,t,r,n,a){let s=t[t.length-1],[i,o]=[e.length/s,s],l=nR.util.getTypedArrayFromDType(r,i*n),u=nR.util.getTypedArrayFromDType("int32",i*n);for(let t=0;t<i;t++){let r=t*o,s=e.subarray(r,r+o),i=Array(s.length);s.forEach((e,t)=>i[t]={value:e,index:t}),n<i.length&&(!function e(t,r,n=0,a=t.length-1){for(;a>n;){if(a-n>600){let s=a-n+1,i=r-n+1,o=Math.log(s),l=.5*Math.exp(2*o/3),u=.5*Math.sqrt(o*l*(s-l)/s)*Math.sign(i-s/2),h=Math.max(n,Math.floor(r-i*l/s+u)),d=Math.min(a,Math.floor(r+(s-i)*l/s+u));e(t,r,h,d)}let s=t[r],i=n,o=a;for(nR.util.swap(t,n,r),cr(t[a],s)>0&&nR.util.swap(t,n,a);i<o;){for(nR.util.swap(t,i,o),i++,o--;0>cr(t[i],s);)i+=1;for(;cr(t[o],s)>0;)o-=1}0===cr(t[n],s)?nR.util.swap(t,n,o):(o+=1,nR.util.swap(t,o,a)),o<=r&&(n=o+1),r<=o&&(a=o-1)}}(i,n),i=i.slice(0,n)),a&&i.sort(cr);let h=t*n,d=l.subarray(h,h+n),p=u.subarray(h,h+n);for(let e=0;e<n;e++)d[e]=i[e].value,p[e]=i[e].index}let h=t.slice();return h[h.length-1]=n,[(0,h5.buffer)(h,r,l),(0,h5.buffer)(h,"int32",u)]}function ca(e,t,r,n){let a=nR.util.parseAxisParam(t,r)[0],s=[1,r[0],1];for(let e=0;e<a;e++)s[0]*=r[e];s[1]=r[a];for(let e=a+1;e<r.length;e++)s[2]*=r[e];let i=new Map,o=new Int32Array(r[a]),l=new rm.TensorBuffer(s,n,e),u=[],h=1===s[0]&&1===s[2];for(let t=0;t<r[a];t++){let r;if(h)r=e[t].toString();else{let e=[];for(let r=0;r<s[0];r++)for(let n=0;n<s[2];n++)e.push(l.get(r,t,n));r=e.join(",")}let n=i.get(r);if(null!=n)o[t]=n;else{let e=i.size;i.set(r,e),o[t]=e,u.push(t)}}let d=s.slice();d[1]=i.size;let p=new rm.TensorBuffer(d,n);u.forEach((e,t)=>{for(let r=0;r<s[0];r++)for(let n=0;n<s[2];n++)p.set(l.get(r,e,n),r,t,n)});let c=r.slice();return c[a]=d[1],{outputValues:p.values,outputShape:c,indices:o}}e.s(["addImpl",0,dx,"bincountImpl",0,dw,"bincountReduceImpl",0,dC,"bitwiseAndImpl",0,dI,"castImpl",0,dp,"ceilImpl",0,dR,"concatImpl",0,dF,"equalImpl",0,dD,"expImpl",0,dz,"expm1Impl",0,dP,"floorDivImpl",0,dH,"floorImpl",0,dG,"gatherNdImpl",0,dX,"gatherV2Impl",0,dK,"greaterEqualImpl",0,dQ,"greaterImpl",0,dY,"lessEqualImpl",0,d5,"lessImpl",0,d2,"linSpaceImpl",0,d7,"logImpl",0,d9,"maxImpl",0,pr,"maximumImpl",0,pn,"minimumImpl",0,pi,"multiplyImpl",0,pu,"negImpl",0,pc,"notEqualImpl",0,pm,"prodImpl",0,pC,"raggedGatherImpl",0,pS,"raggedRangeImpl",0,pT,"raggedTensorToTensorImpl",0,pE,"rangeImpl",0,pF,"rsqrtImpl",0,pD,"scatterImpl",0,pz,"sigmoidImpl",0,p_,"simpleAbsImpl",0,dt,"sliceImpl",0,pW,"sparseFillEmptyRowsImpl",0,pV,"sparseReshapeImpl",0,pH,"sparseSegmentReductionImpl",0,pq,"sqrtImpl",0,pj,"squaredDifferenceImpl",0,pY,"staticRegexReplaceImpl",0,pQ,"stridedSliceImpl",0,p2,"stringNGramsImpl",0,p4,"stringSplitImpl",0,p5,"stringToHashBucketFastImpl",0,p6,"subImpl",0,p8,"tileImpl",0,ct,"topKImpl",0,cn,"transposeImpl",0,pb,"uniqueImpl",0,ca],62710);var cs=e.i(62710);(0,n$.registerBackend)("cpu",()=>new de,1),e.s([],60131);let ci=dN(x.Elu,e=>e>=0?e:Math.exp(e)-1),co={kernelName:x.Elu,backendName:"cpu",kernelFunc:ci};function cl(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{alpha:s}=n;h7([a],"leakyRelu");let i=nR.util.sizeFromShape(a.shape),o=r.data.get(a.dataId).values,l=nR.util.getTypedArrayFromDType("float32",i);for(let e=0;e<o.length;e++)l[e]=o[e]<0?s*o[e]:o[e];return r.makeTensorInfo(a.shape,"float32",l)}let cu={kernelName:x.LeakyRelu,backendName:"cpu",kernelFunc:cl},ch=dn((e,t)=>e<0?t*e:e);function cd(e){let{inputs:t,backend:r}=e,{x:n,alpha:a}=t;h7([n,a],"prelu");let s=r.data.get(n.dataId).values,i=r.data.get(a.dataId).values,[o,l]=ch(n.shape,a.shape,s,i,"float32");return r.makeTensorInfo(l,"float32",o)}let cp={kernelName:x.Prelu,backendName:"cpu",kernelFunc:cd},cc=dN(x.Relu,e=>Math.max(0,e)),cf={kernelName:x.Relu,backendName:"cpu",kernelFunc:cc},cm=dN(x.Relu6,e=>Math.min(Math.max(0,e),6)),cg={kernelName:x.Relu6,backendName:"cpu",kernelFunc:cm};function cx(e,t,r,n,a){if("linear"===r)return dl({inputs:{x:t},backend:e});if("relu"===r)return cc({inputs:{x:t},backend:e});if("elu"===r)return ci({inputs:{x:t},backend:e});if("relu6"===r)return cm({inputs:{x:t},backend:e});if("prelu"===r)return cd({inputs:{x:t,alpha:n},backend:e});else if("leakyrelu"===r)return cl({inputs:{x:t},backend:e,attrs:{alpha:a}});else if("sigmoid"===r)return pM({inputs:{x:t},backend:e});throw Error(`Activation ${r} has not been implemented for the CPU backend.`)}var cy=A;function cb(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{shape:s}=n,i=nR.util.sizeFromShape(a.shape),o=nR.util.inferFromImplicitShape(s,i),l=nR.util.sizeFromShape(o);nR.util.assert(i===l,()=>`The new shape (${o}) has ${l} elements and the old shape (${a.shape}) has ${i} elements. The new shape and old shape must have the same number of elements.`),r.incRef(a.dataId);let u=r.data.get(a.dataId);if(null!=u.complexTensorInfos){let e=u.complexTensorInfos.real,t=u.complexTensorInfos.imag;e.shape=o,t.shape=o}return{dataId:a.dataId,shape:o,dtype:a.dtype}}let cv={kernelName:x.Reshape,backendName:"cpu",kernelFunc:cb};function cw(e){let{inputs:t,backend:r,attrs:n}=e,{a,b:s}=t,{transposeA:i,transposeB:o}=n;h7([a,s],"matMul");let l=a.shape.length,u=s.shape.length,h=i?a.shape[l-2]:a.shape[l-1],d=o?s.shape[u-1]:s.shape[u-2],p=i?a.shape[l-1]:a.shape[l-2],c=o?s.shape[u-2]:s.shape[u-1],f=a.shape.slice(0,-2),m=s.shape.slice(0,-2),g=nR.util.sizeFromShape(f),x=nR.util.sizeFromShape(m),y=cy.assertAndGetBroadcastShape(a.shape.slice(0,-2),s.shape.slice(0,-2)).concat([p,c]);nR.util.assert(h===d,()=>`Error in matMul: inner shapes (${h}) and (${d}) of Tensors with shapes ${a.shape} and ${s.shape} and transposeA=${i} and transposeB=${o} must match.`);let b=cb({inputs:{x:a},backend:r,attrs:{shape:i?[g,h,p]:[g,p,h]}}),v=cb({inputs:{x:s},backend:r,attrs:{shape:o?[x,c,d]:[x,d,c]}}),w=i?b.shape[1]:b.shape[2],C=i?b.shape[2]:b.shape[1],I=o?v.shape[1]:v.shape[2],k=Math.max(g,x),S=r.data.get(b.dataId).values,T=r.data.get(v.dataId).values,N=nR.util.computeStrides(b.shape),$=nR.util.computeStrides(v.shape),[R,A,E]=i?[N[0],1,N[1]]:[N[0],N[1],1],[F,D,O]=o?[1,$[1],$[0]]:[$[1],1,$[0]],L=C*I,z=(0,h5.buffer)([k,C,I],b.dtype),_=z.values,M=r.blockSize;for(let e=0;e<k;e++){let t=e%g,r=e%x;for(let n=0;n<C;n+=M){let a=Math.min(n+M,C);for(let s=0;s<I;s+=M){let i=Math.min(s+M,I);for(let o=0;o<w;o+=M){let l=Math.min(o+M,w);for(let u=n;u<a;u++)for(let n=s;n<i;n++){let a=0;for(let e=o;e<l;e++)a+=S[t*R+u*A+e*E]*T[e*F+n*D+r*O];_[e*L+(u*I+n)]+=a}}}}}return r.disposeIntermediateTensorInfo(b),r.disposeIntermediateTensorInfo(v),r.makeTensorInfo(y,z.dtype,z.values)}let cC={kernelName:x.BatchMatMul,backendName:"cpu",kernelFunc:cw},cI={kernelName:x._FusedMatMul,backendName:"cpu",kernelFunc:function(e){let t,r,n,{inputs:a,backend:s,attrs:i}=e,{a:o,b:l,bias:u,preluActivationWeights:h}=a,{transposeA:d,transposeB:p,activation:c,leakyreluAlpha:f}=i,m=[];for(let e of(t=cw({inputs:{a:o,b:l},attrs:{transposeA:d,transposeB:p},backend:s}),u&&(r=db({inputs:{a:t,b:u},backend:s}),m.push(t),t=r),c&&(n=cx(s,t,c,h,f),m.push(t),t=n),m))s.disposeIntermediateTensorInfo(e);return t}},ck=dN(x.Acos,e=>Math.acos(e)),cS={kernelName:x.Acos,backendName:"cpu",kernelFunc:ck},cT=dN(x.Acosh,e=>Math.acosh(e)),cN={kernelName:x.Acosh,backendName:"cpu",kernelFunc:cT},c$={kernelName:x.AddN,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e;h7(t,"addN");let n=t.map(e=>r.data.get(e.dataId).values),a=(0,h5.buffer)(t[0].shape,t[0].dtype),s=a.values;for(let e=0;e<t.length;e++){let t=n[e];for(let e=0;e<s.length;e++)s[e]+=t[e]}return r.makeTensorInfo(a.shape,a.dtype,a.values)}};var tN=tN;let cR={kernelName:x.All,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,keepDims:i}=n;h7(a,"all");let o=nR.util.parseAxisParam(s,a.shape),l=o,u=tN.getAxesPermutation(l,a.shape.length),h=a;null!=u&&(h=pv({inputs:{x:a},backend:r,attrs:{perm:u}}),l=tN.getInnerMostAxes(l.length,a.shape.length)),tN.assertAxesAreInnerMostDims("all",l,h.shape.length);let[d,p]=tN.computeOutAndReduceShapes(h.shape,l),c=nR.util.sizeFromShape(p),f=nR.util.makeZerosTypedArray(nR.util.sizeFromShape(d),h.dtype),m=r.data.get(h.dataId).values;for(let e=0;e<f.length;++e){let t=e*c,r=m[t];for(let e=0;e<c;++e){let n=m[t+e];r=r&&n}f[e]=r}null!=u&&r.disposeIntermediateTensorInfo(h);let g=r.makeTensorInfo(d,h.dtype,f);if(i){let e=cb({inputs:{x:g},backend:r,attrs:{shape:tN.expandShapeToKeepDim(d,o)}});return r.disposeIntermediateTensorInfo(g),e}return g}};var tN=tN;let cA={kernelName:x.Any,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,keepDims:i}=n;h7(a,"any");let o=nR.util.parseAxisParam(s,a.shape),l=o,u=tN.getAxesPermutation(l,a.shape.length),h=a;null!=u&&(h=pv({inputs:{x:a},backend:r,attrs:{perm:u}}),l=tN.getInnerMostAxes(l.length,a.shape.length)),tN.assertAxesAreInnerMostDims("any",l,h.shape.length);let[d,p]=tN.computeOutAndReduceShapes(h.shape,l),c=nR.util.sizeFromShape(p),f=nR.util.makeZerosTypedArray(nR.util.sizeFromShape(d),h.dtype),m=r.data.get(h.dataId).values;for(let e=0;e<f.length;++e){let t=e*c,r=m[t];for(let e=0;e<c;++e){let n=m[t+e];r=r||n}f[e]=r}null!=u&&r.disposeIntermediateTensorInfo(h);let g=r.makeTensorInfo(d,h.dtype,f);if(i){let e=cb({inputs:{x:g},backend:r,attrs:{shape:tN.expandShapeToKeepDim(d,o)}});return r.disposeIntermediateTensorInfo(g),e}return g}};var tN=tN;let cE={kernelName:x.ArgMax,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s}=n;h7(a,"argMax");let i=nR.util.parseAxisParam(s,a.shape),o=tN.getAxesPermutation(i,a.shape.length),l=a,u=[];null!=o&&(u.push(l=pv({inputs:{x:a},backend:r,attrs:{perm:o}})),i=tN.getInnerMostAxes(i.length,l.shape.length)),i=[i[0]],tN.assertAxesAreInnerMostDims("argMax",i,l.shape.length);let[h,d]=tN.computeOutAndReduceShapes(l.shape,i),p=nR.util.sizeFromShape(h),c=nR.util.makeZerosTypedArray(p,"int32"),f=nR.util.sizeFromShape(d),m=r.data.get(l.dataId).values;for(let e=0;e<c.length;++e){let t=e*f,r=m[t],n=0;for(let e=0;e<f;++e){let a=m[t+e];a>r&&(r=a,n=e)}c[e]=n}return u.forEach(e=>r.disposeIntermediateTensorInfo(e)),r.makeTensorInfo(h,"int32",c)}};var tN=tN;let cF={kernelName:x.ArgMin,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s}=n;h7(a,"argMin");let i=nR.util.parseAxisParam(s,a.shape),o=tN.getAxesPermutation(i,a.shape.length),l=a,u=[];null!=o&&(u.push(l=pv({inputs:{x:a},backend:r,attrs:{perm:o}})),i=tN.getInnerMostAxes(i.length,l.shape.length)),i=[i[0]],tN.assertAxesAreInnerMostDims("argMin",i,l.shape.length);let[h,d]=tN.computeOutAndReduceShapes(l.shape,i),p=nR.util.sizeFromShape(h),c=nR.util.makeZerosTypedArray(p,"int32"),f=nR.util.sizeFromShape(d),m=r.data.get(l.dataId).values;for(let e=0;e<c.length;++e){let t=e*f,r=m[t],n=0;for(let e=0;e<f;++e){let a=m[t+e];a<r&&(r=a,n=e)}c[e]=n}return u.forEach(e=>r.disposeIntermediateTensorInfo(e)),r.makeTensorInfo(h,"int32",c)}},cD=dN(x.Asin,e=>Math.asin(e)),cO={kernelName:x.Asin,backendName:"cpu",kernelFunc:cD},cL=dN(x.Asinh,e=>Math.asinh(e)),cz={kernelName:x.Asinh,backendName:"cpu",kernelFunc:cL},c_=dN(x.Atan,e=>Math.atan(e)),cM={kernelName:x.Atan,backendName:"cpu",kernelFunc:c_},cP=dn((e,t)=>Math.atan2(e,t)),cB=dm(x.Atan2,cP),cW={kernelName:x.Atan2,backendName:"cpu",kernelFunc:cB},cG=dN(x.Atanh,e=>Math.atanh(e)),cU={kernelName:x.Atanh,backendName:"cpu",kernelFunc:cG};var tN=tN;function cV(e,t,r,n,a,s){let i=a.strideHeight,o=a.strideWidth,l=a.dilationHeight,u=a.dilationWidth,h=a.effectiveFilterHeight,d=a.effectiveFilterWidth,p=a.padInfo.top,c=a.padInfo.left,f="max"===s?-1/0:1/0,m=(0,h5.buffer)(a.outShape,r),g=m.values,x=a.outShape[1]*a.outShape[2]*a.outShape[3],y=a.outShape[2]*a.outShape[3],b=a.outShape[3];for(let t=0;t<a.batchSize;++t){let r=t*x,m=t*n[0];for(let t=0;t<a.inChannels;++t)for(let x=0;x<a.outHeight;++x){let v=x*i-p,w=Math.max(0,v),C=Math.min(a.inHeight,h+v),I=r+x*y;for(let r=0;r<a.outWidth;++r){let i=r*o-c,h=Math.max(0,i),p=Math.min(a.inWidth,d+i),x=f,y=0,v=0;for(let r=w;r<C;r+=l){let a=m+r*n[1];for(let r=h;r<p;r+=u){let i=e[a+r*n[2]+t];"max"===s&&i>x?x=i:"avg"===s&&(y+=i,v++)}if(isNaN(x))break}g[I+r*b+t]="avg"===s?y/v:x}}}return m}function cH(e,t,r,n,a=!1,s=!1){let i=(0,h5.buffer)(n.outShape,"int32"),o=n.strideHeight,l=n.strideWidth,u=n.dilationHeight,h=n.dilationWidth,d=n.effectiveFilterHeight,p=n.effectiveFilterWidth,c=n.padInfo.top,f=n.padInfo.left,m=(0,h5.buffer)(t,r,e);for(let e=0;e<n.batchSize;++e)for(let t=0;t<n.inChannels;++t)for(let r=0;r<n.outHeight;++r){let g=r*o-c,x=g;for(;x<0;)x+=u;let y=Math.min(n.inHeight,d+g);for(let o=0;o<n.outWidth;++o){let d=o*l-f,c=d;for(;c<0;)c+=h;let b=Math.min(n.inWidth,p+d),v=-1/0,w=-1;for(let r=x;r<y;r+=u){let i=r-g;for(let o=c;o<b;o+=h){let l=o-d,u=m.get(e,r,o,t);u>v&&(v=u,w=a?s?((e*n.inHeight+r)*n.inWidth+o)*n.inChannels+t:(r*n.inWidth+o)*n.inChannels+t:i*p+l)}}i.set(w,e,r,o,t)}}return i}function cq(e,t,r,n,a,s){let i=a.strideDepth,o=a.strideHeight,l=a.strideWidth,u=a.dilationDepth,h=a.dilationHeight,d=a.dilationWidth,p=a.effectiveFilterDepth,c=a.effectiveFilterHeight,f=a.effectiveFilterWidth,m=a.padInfo.front,g=a.padInfo.top,x=a.padInfo.left,y="max"===s?-1/0:1/0,b=(0,h5.buffer)(a.outShape,r),v=b.values,w=a.outShape[1]*a.outShape[2]*a.outShape[3]*a.outShape[4],C=a.outShape[2]*a.outShape[3]*a.outShape[4],I=a.outShape[3]*a.outShape[4],k=a.outShape[4];for(let t=0;t<a.batchSize;++t){let r=t*w,b=t*n[0];for(let t=0;t<a.inChannels;++t)for(let w=0;w<a.outDepth;++w){let S=w*i-m,T=S;for(;T<0;)T+=u;let N=Math.min(a.inDepth,p+S),$=r+w*C;for(let r=0;r<a.outHeight;++r){let i=r*o-g,p=i;for(;p<0;)p+=h;let m=Math.min(a.inHeight,c+i),w=$+r*I;for(let r=0;r<a.outWidth;++r){let i=r*l-x,o=i;for(;o<0;)o+=d;let c=Math.min(a.inWidth,f+i),g=w+r*k,C=y,I=0,S=0;for(let r=T;r<N;r+=u){let a=b+r*n[1];for(let r=p;r<m;r+=h){let i=a+r*n[2];for(let r=o;r<c;r+=d){let a=e[i+r*n[3]+t];if("max"===s&&a>C?C=a:"avg"===s&&(I+=a,S++),isNaN(C))break}if(isNaN(C))break}if(isNaN(C))break}v[g+t]="avg"===s?I/Math.max(S,1):C}}}}return b}let cj={kernelName:x.AvgPool,backendName:"cpu",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r;h7(s,"avgPool");let{filterSize:i,strides:o,pad:l,dimRoundingMode:u}=a;nR.util.assert(tN.eitherStridesOrDilationsAreOne(o,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let h=tN.computePool2DInfo(s.shape,i,o,1,l,u);if(1===h.filterWidth&&1===h.filterHeight&&nR.util.arraysEqual(h.inShape,h.outShape))t=dl({inputs:{x:s},backend:n});else{let e=n.data.get(s.dataId).values,r=nR.util.computeStrides(s.shape),a=cV(e,s.shape,s.dtype,r,h,"avg");t=n.makeTensorInfo(h.outShape,s.dtype,a.values)}return t}};var tN=tN;let cX={kernelName:x.AvgPool3D,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{filterSize:s,strides:i,pad:o,dimRoundingMode:l,dataFormat:u}=n;h7(a,"avgPool3d");let h=tN.computePool3DInfo(a.shape,s,i,1,o,l,u),d=cq(r.data.get(a.dataId).values,a.shape,a.dtype,nR.util.computeStrides(a.shape),h,"avg");return r.makeTensorInfo(d.shape,"float32",d.values)}};var tN=tN;let cK={kernelName:x.AvgPool3DGrad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,input:s}=t,{filterSize:i,strides:o,pad:l,dimRoundingMode:u}=n;h7([a,s],"avgPool3DGrad");let h=tN.computePool3DInfo(s.shape,i,o,1,l,u),d=h.strideDepth,p=h.strideHeight,c=h.strideWidth,f=h.filterDepth,m=h.filterHeight,g=h.filterWidth,x=h.dilationDepth,y=h.dilationHeight,b=h.dilationWidth,v=h.effectiveFilterDepth,w=h.effectiveFilterHeight,C=h.effectiveFilterWidth,I=v-1-h.padInfo.front,k=C-1-h.padInfo.left,S=w-1-h.padInfo.top,T=(0,h5.buffer)(s.shape,"float32"),N=1/(f*m*g),$=r.bufferSync(a);for(let e=0;e<h.batchSize;++e)for(let t=0;t<h.inChannels;++t)for(let r=0;r<h.inDepth;++r)for(let n=0;n<h.inHeight;++n)for(let a=0;a<h.inWidth;++a){let s=r-I,i=n-S,o=a-k,l=0;for(let r=0;r<v;r+=x){let n=(s+r)/d;if(!(n<0)&&!(n>=h.outDepth)&&Math.floor(n)===n)for(let r=0;r<w;r+=y){let a=(i+r)/p;if(!(a<0)&&!(a>=h.outHeight)&&Math.floor(a)===a)for(let r=0;r<C;r+=b){let s=(o+r)/c;s<0||s>=h.outWidth||Math.floor(s)!==s||(l+=$.get(e,n,a,s,t))}}}T.set(l*N,e,r,n,a,t)}return r.makeTensorInfo(T.shape,T.dtype,T.values)}};var tN=tN;let cY={kernelName:x.AvgPoolGrad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,input:s}=t;h7([a,s],"avgPoolGrad");let{filterSize:i,strides:o,pad:l}=n,u=tN.computePool2DInfo(s.shape,i,o,1,l),h=u.strideHeight,d=u.strideWidth,p=u.filterHeight,c=u.filterWidth,f=u.dilationHeight,m=u.dilationWidth,g=u.effectiveFilterHeight,x=u.effectiveFilterWidth,y=x-1-u.padInfo.left,b=g-1-u.padInfo.top,v=(0,h5.buffer)(s.shape,"float32"),w=1/(p*c),C=r.data.get(a.dataId).values,I=(0,h5.buffer)(a.shape,"float32",C);for(let e=0;e<u.batchSize;++e)for(let t=0;t<u.inChannels;++t)for(let r=0;r<u.inHeight;++r)for(let n=0;n<u.inWidth;++n){let a=r-b,s=n-y,i=0;for(let r=0;r<g;r+=f){let n=(a+r)/h;if(!(n<0)&&!(n>=u.outHeight)&&Math.floor(n)===n)for(let r=0;r<x;r+=m){let a=(s+r)/d;a<0||a>=u.outWidth||Math.floor(a)!==a||(i+=I.get(e,n,a,t))}}v.set(i*w,e,r,n,t)}return r.makeTensorInfo(v.shape,v.dtype,v.values)}},cZ={kernelName:x.FusedBatchNorm,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,scale:s,offset:i,mean:o,variance:l}=t;nR.util.assert(o.shape.length===l.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),nR.util.assert(null==i||o.shape.length===i.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),nR.util.assert(null==s||o.shape.length===s.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks."),h7([a,o,l,s,i],"batchNorm");let{varianceEpsilon:u}=n;null==u&&(u=.001);let h=r.data.get(a.dataId).values,d=r.data.get(o.dataId).values,p=r.data.get(l.dataId).values,c=s?r.data.get(s.dataId).values:new Float32Array([1]),f=i?r.data.get(i.dataId).values:new Float32Array([0]),m=new Float32Array(h.length),g=f.length,x=c.length,y=p.length,b=d.length,v=0,w=0,C=0,I=0;for(let e=0;e<h.length;++e)m[e]=f[v++]+(h[e]-d[w++])*c[C++]/Math.sqrt(p[I++]+u),v>=g&&(v=0),w>=b&&(w=0),C>=x&&(C=0),I>=y&&(I=0);return r.makeTensorInfo(a.shape,a.dtype,m)}};var tN=tN;let cJ={kernelName:x.BatchToSpaceND,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{blockShape:s,crops:i}=n;h7([a],"batchToSpaceND");let o=s.reduce((e,t)=>e*t),l=tN.getReshaped(a.shape,s,o),u=tN.getPermuted(l.length,s.length),h=tN.getReshapedPermuted(a.shape,s,o),d=tN.getSliceBeginCoords(i,s.length),p=tN.getSliceSize(h,i,s.length),c=cb({inputs:{x:a},backend:r,attrs:{shape:l}}),f=pv({inputs:{x:c},backend:r,attrs:{perm:u}}),m=cb({inputs:{x:f},backend:r,attrs:{shape:h}}),g=pG({inputs:{x:m},backend:r,attrs:{begin:d,size:p}});return r.disposeIntermediateTensorInfo(c),r.disposeIntermediateTensorInfo(f),r.disposeIntermediateTensorInfo(m),g}},cQ={kernelName:x.Bincount,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,weights:s}=t,{size:i}=n,o=dw(r.data.get(a.dataId).values,r.data.get(s.dataId).values,s.dtype,s.shape,i);return r.makeTensorInfo([i],s.dtype,o)}};var tN=tN;let c0={kernelName:x.BroadcastArgs,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{s0:n,s1:a}=t,s=r.data.get(n.dataId).values,i=r.data.get(a.dataId).values,o=tN.assertAndGetBroadcastShape(Array.from(s),Array.from(i));return r.makeTensorInfo([o.length],"int32",Int32Array.from(o))}},c1=dN(x.ClipByValue,(e,t)=>e>t.clipValueMax?t.clipValueMax:e<t.clipValueMin?t.clipValueMin:e),c2={kernelName:x.ClipByValue,backendName:"cpu",kernelFunc:c1},c3={kernelName:x.ComplexAbs,backendName:"cpu",kernelFunc:e=>{let{x:t}=e.inputs,r=e.backend,n=new Float32Array(nR.util.sizeFromShape(t.shape)),a=r.data.get(t.dataId),s=a.complexTensorInfos.real,i=a.complexTensorInfos.imag,o=r.data.get(s.dataId).values,l=r.data.get(i.dataId).values;for(let e=0;e<o.length;e++){let t=o[e],r=l[e];n[e]=Math.hypot(t,r)}return r.makeOutput(n,t.shape,"float32")}};var tN=tN;function c4(e){let{inputs:t,backend:r}=e,{input:n}=t,a=r.data.get(n.dataId).complexTensorInfos.imag,s=r.data.get(a.dataId).values;return r.makeTensorInfo(a.shape,a.dtype,s)}let c5={kernelName:x.Imag,backendName:"cpu",kernelFunc:c4};function c6(e){let{inputs:t,backend:r,attrs:n}=e,{axis:a}=n,s=nR.util.parseAxisParam(a,t[0].shape)[0],i=t.map(e=>e.shape);tN.assertParamsConsistent(i,s);let o=tN.computeOutShape(t.map(e=>e.shape),s);if(0===nR.util.sizeFromShape(o))return r.makeTensorInfo(o,t[0].dtype,[]);let l=t.filter(e=>nR.util.sizeFromShape(e.shape)>0);if(1===l.length)return dl({inputs:{x:l[0]},backend:r});if("complex64"===l[0].dtype){let e=l.map(e=>dh({inputs:{input:e},backend:r})),t=l.map(e=>c4({inputs:{input:e},backend:r})),n=c6({inputs:e,backend:r,attrs:{axis:s}}),a=c6({inputs:t,backend:r,attrs:{axis:s}}),i=da({inputs:{real:n,imag:a},backend:r});return e.forEach(e=>r.disposeIntermediateTensorInfo(e)),t.forEach(e=>r.disposeIntermediateTensorInfo(e)),r.disposeIntermediateTensorInfo(n),r.disposeIntermediateTensorInfo(a),i}let u=l.map(e=>{let t=nR.util.sizeFromShape(e.shape.slice(s));return cb({inputs:{x:e},backend:r,attrs:{shape:[-1,t]}})}),h=u.map(e=>({vals:r.data.get(e.dataId).values,shape:e.shape}));o=tN.computeOutShape(u.map(e=>e.shape),1);let d=1===u[0].shape[0],p=dF(h,o,t[0].dtype,d),c=tN.computeOutShape(l.map(e=>e.shape),s),f=r.makeTensorInfo(c,t[0].dtype,p);return u.forEach(e=>r.disposeIntermediateTensorInfo(e)),f}let c8={kernelName:x.Concat,backendName:"cpu",kernelFunc:c6};var tN=tN;function c7(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,filter:s}=t,{strides:i,pad:o,dataFormat:l,dilations:u,dimRoundingMode:h}=n;h7([a,s],"conv2d");let d=tN.convertConv2DDataFormat(l),p=tN.computeConv2DInfo(a.shape,s.shape,i,u,o,h,!1,d),c=p.filterHeight,f=p.filterWidth,m=p.dilationHeight,g=p.dilationWidth,x=p.padInfo.left,y=p.padInfo.top,b="channelsLast"===p.dataFormat,v=new rm.TensorBuffer(p.outShape,a.dtype),w=nR.util.computeStrides(a.shape),C=nR.util.computeStrides(s.shape),I=w[0],k=b?w[1]:w[2],S=b?w[2]:1,T=b?1:w[1],N=v.strides[0],$=b?v.strides[1]:v.strides[2],R=b?v.strides[2]:1,A=b?1:v.strides[1],E=r.data.get(a.dataId).values,F=r.data.get(s.dataId).values,D=v.values;for(let e=0;e<p.batchSize;++e){let t=e*I,r=e*N;for(let e=0;e<p.outHeight;++e){let n=r+e*$,a=e*p.strideHeight-y;for(let e=0;e<c;++e){let r=a+e*m;if(r<0||r>=p.inHeight)continue;let s=e*C[0],i=t+r*k;for(let e=0;e<p.outWidth;++e){let t=n+e*R,r=e*p.strideWidth-x;for(let e=0;e<f;++e){let n=r+e*g;if(n<0||n>=p.inWidth)continue;let a=s+e*C[1],o=i+n*S,l=a;for(let e=0;e<p.inChannels;++e){let r=E[o+e*T];for(let e=0;e<p.outChannels;++e)D[t+e*A]+=r*F[l+e];l+=p.outChannels}}}}}}return r.makeTensorInfo(v.shape,v.dtype,D)}let c9={kernelName:x.Conv2D,backendName:"cpu",kernelFunc:c7};var tN=tN;let fe={kernelName:x.Conv2DBackpropFilter,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,dy:s}=t,{strides:i,pad:o,dataFormat:l,dimRoundingMode:u,filterShape:h}=n;h7([a,s],"conv2dBackpropFilter");let d=tN.convertConv2DDataFormat(l),p=tN.computeConv2DInfo(a.shape,h,i,1,o,u,!1,d),{strideHeight:c,strideWidth:f,filterHeight:m,filterWidth:g}=p,x="channelsLast"===p.dataFormat,y=new rm.TensorBuffer(p.filterShape,"float32"),b=p.padInfo.left,v=p.padInfo.top,w=r.data.get(a.dataId).values,C=r.data.get(s.dataId).values,I=new rm.TensorBuffer(a.shape,a.dtype,w),k=new rm.TensorBuffer(s.shape,s.dtype,C);for(let e=0;e<m;++e){let t=Math.max(0,Math.ceil((v-e)/c)),r=Math.min(p.outHeight,(p.inHeight+v-e)/c);for(let n=0;n<g;++n){let a=Math.max(0,Math.ceil((b-n)/f)),s=Math.min(p.outWidth,(p.inWidth+b-n)/f);for(let i=0;i<p.inChannels;++i)for(let o=0;o<p.outChannels;++o){let l=0;for(let u=0;u<p.batchSize;++u)for(let h=t;h<r;++h){let t=e+h*c-v;for(let e=a;e<s;++e){let r=n+e*f-b;x?l+=I.get(u,t,r,i)*k.get(u,h,e,o):l+=I.get(u,i,t,r)*k.get(u,o,h,e)}}y.set(l,e,n,i,o)}}}return r.makeTensorInfo(y.shape,y.dtype,y.values)}};var tN=tN;let ft={kernelName:x.Conv2DBackpropInput,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,filter:s}=t,{inputShape:i,strides:o,pad:l,dataFormat:u,dimRoundingMode:h}=n;h7([a,s],"conv2dBackpropInput");let d=nR.util.computeStrides(s.shape),p=nR.util.computeStrides(a.shape),c=tN.convertConv2DDataFormat(u),f=tN.computeConv2DInfo(i,s.shape,o,1,l,h,!1,c),m=new rm.TensorBuffer(f.inShape,"float32"),g=m.values,x=r.data.get(a.dataId).values,y=r.data.get(s.dataId).values,[b,v,w]=d,{batchSize:C,filterHeight:I,filterWidth:k,inChannels:S,inHeight:T,inWidth:N,outChannels:$,outHeight:R,outWidth:A,strideHeight:E,strideWidth:F}=f;c=f.dataFormat;let D=I-1-f.padInfo.top,O=k-1-f.padInfo.left,L="channelsLast"===c,z=m.strides[0],_=L?m.strides[1]:m.strides[2],M=L?m.strides[2]:1,P=L?1:m.strides[1],B=p[0],W=L?p[1]:p[2],G=L?p[2]:1,U=L?1:p[1];for(let e=0;e<C;++e)for(let t=0;t<S;++t)for(let r=0;r<T;++r){let n=r-D,a=Math.max(0,Math.ceil(n/E)),s=Math.min(R,(I+n)/E);for(let i=0;i<N;++i){let o=i-O,l=Math.max(0,Math.ceil(o/F)),u=Math.min(A,(k+o)/F),h=0;for(let r=a;r<s;++r){let a=r*E-n;for(let n=l;n<u;++n){let s=n*F-o,i=B*e+W*r+G*n,l=b*(I-1-a)+v*(k-1-s)+w*t;for(let e=0;e<$;++e)h+=x[i+U*e]*y[l+e]}}g[z*e+_*r+M*i+P*t]=h}}return r.makeTensorInfo(m.shape,m.dtype,m.values)}};var tN=tN;let fr={kernelName:x.Conv3D,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,filter:s}=t,{strides:i,pad:o,dilations:l}=n;h7([a,s],"conv3d");let u=tN.computeConv3DInfo(a.shape,s.shape,i,l,o),{filterDepth:h,filterHeight:d,filterWidth:p,dilationDepth:c,dilationHeight:f,dilationWidth:m,padInfo:g}=u,x=g.front,y=g.left,b=g.top,v=new rm.TensorBuffer(u.outShape,a.dtype),w=r.data.get(a.dataId).values,C=r.data.get(s.dataId).values,I=v.values,k=nR.util.computeStrides(a.shape),S=nR.util.computeStrides(s.shape);for(let e=0;e<u.batchSize;++e){let t=e*k[0],r=e*v.strides[0];for(let e=0;e<u.outDepth;++e){let n=r+e*v.strides[1],a=e*u.strideDepth-x;for(let e=0;e<h;++e){let r=a+e*c;if(r<0||r>=u.inDepth)continue;let s=e*S[0],i=t+r*k[1];for(let e=0;e<u.outHeight;++e){let t=n+e*v.strides[2],r=e*u.strideHeight-b;for(let e=0;e<d;++e){let n=r+e*f;if(n<0||n>=u.inHeight)continue;let a=s+e*S[1],o=i+n*k[2];for(let e=0;e<u.outWidth;++e){let r=t+e*u.outChannels,n=e*u.strideWidth-y;for(let e=0;e<p;++e){let t=n+e*m;if(t<0||t>=u.inWidth)continue;let s=a+e*S[2],i=o+t*u.inChannels,l=s;for(let e=0;e<u.inChannels;++e){let t=w[i+e];for(let e=0;e<u.outChannels;++e)I[r+e]+=t*C[l+e];l+=u.outChannels}}}}}}}}return r.makeTensorInfo(v.shape,v.dtype,v.values)}};var tN=tN;let fn={kernelName:x.Conv3DBackpropFilterV2,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,dy:s}=t,{strides:i,pad:o,filterShape:l}=n;h7([a,s],"conv3dBackpropFilterV2");let u=nR.util.computeStrides(a.shape),h=nR.util.computeStrides(s.shape),d=tN.computeConv3DInfo(a.shape,l,i,1,o),p=d.strideDepth,c=d.strideHeight,f=d.strideWidth,m=d.filterDepth,g=d.filterHeight,x=d.filterWidth,y=new rm.TensorBuffer(d.filterShape,"float32"),b=y.values,[v,w,C,I]=y.strides,k=r.data.get(s.dataId).values,[S,T,N,$]=h,R=r.data.get(a.dataId).values,[A,E,F,D]=u,O=d.padInfo.front,L=d.padInfo.left,z=d.padInfo.top;for(let e=0;e<m;++e){let t=Math.max(0,Math.ceil((O-e)/p)),r=Math.min(d.outDepth,(d.inDepth+O-e)/p),n=e*v;for(let a=0;a<g;++a){let s=Math.max(0,Math.ceil((z-a)/c)),i=Math.min(d.outHeight,(d.inHeight+z-a)/c),o=a*w+n;for(let n=0;n<x;++n){let l=Math.max(0,Math.ceil((L-n)/f)),u=Math.min(d.outWidth,(d.inWidth+L-n)/f),h=n*C+o;for(let o=0;o<d.inChannels;++o){let m=o*I+h;for(let h=0;h<d.outChannels;++h){let g=0;for(let m=0;m<d.batchSize;++m){let d=m*A,x=m*S;for(let m=t;m<r;++m){let t=(e+m*p-O)*E+d,r=m*T+x;for(let e=s;e<i;++e){let s=(a+e*c-z)*F+t,i=e*N+r;for(let e=l;e<u;++e){let t=(n+e*f-L)*D+s,r=e*$+i;g+=R[t+o]*k[r+h]}}}}b[m+h]=g}}}}}return r.makeTensorInfo(y.shape,y.dtype,y.values)}};var tN=tN;let fa={kernelName:x.Conv3DBackpropInputV2,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,filter:s}=t,{pad:i,strides:o,inputShape:l}=n;h7([a],"conv3dBackpropInputV2");let u=nR.util.computeStrides(a.shape),h=nR.util.computeStrides(s.shape),d=tN.computeConv3DInfo(l,s.shape,o,1,i),p=new rm.TensorBuffer(d.inShape,"float32"),c=p.values,[f,m,g,x]=p.strides,y=r.data.get(a.dataId).values,[b,v,w,C]=u,I=r.data.get(s.dataId).values,[k,S,T,N]=h,{batchSize:$,filterDepth:R,filterHeight:A,filterWidth:E,inChannels:F,inDepth:D,inHeight:O,inWidth:L,outChannels:z,outDepth:_,outHeight:M,outWidth:P,strideDepth:B,strideHeight:W,strideWidth:G}=d,U=R-1-d.padInfo.front,V=A-1-d.padInfo.top,H=E-1-d.padInfo.left;for(let e=0;e<$;++e)for(let t=0;t<F;++t)for(let r=0;r<D;++r){let n=r-U,a=Math.max(0,Math.ceil(n/B)),s=Math.min(_,(R+n)/B);for(let i=0;i<O;++i){let o=i-V,l=Math.max(0,Math.ceil(o/W)),u=Math.min(M,(A+o)/W);for(let h=0;h<L;++h){let d=h-H,p=Math.max(0,Math.ceil(d/G)),$=Math.min(P,(E+d)/G),F=0;for(let r=a;r<s;++r){let a=r*B-n;for(let n=l;n<u;++n){let s=n*W-o;for(let i=p;i<$;++i){let o=i*G-d,l=b*e+v*r+w*n+C*i,u=k*(R-1-a)+S*(A-1-s)+T*(E-1-o)+N*t;for(let e=0;e<z;++e)F+=y[l+e]*I[u+e]}}}c[f*e+m*r+g*i+x*h+t]=F}}}return r.makeTensorInfo(p.shape,p.dtype,p.values)}},fs=dN(x.Cos,e=>Math.cos(e)),fi={kernelName:x.Cos,backendName:"cpu",kernelFunc:fs},fo=dN(x.Cosh,e=>Math.cosh(e)),fl={kernelName:x.Cosh,backendName:"cpu",kernelFunc:fo},fu={kernelName:x.CropAndResize,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{image:a,boxes:s,boxInd:i}=t,{cropSize:o,method:l,extrapolationValue:u}=n,[h,d,p,c]=a.shape,f=s.shape[0],[m,g]=o,x=(0,h5.buffer)([f,m,g,c],"float32"),y=r.data.get(s.dataId).values,b=r.data.get(i.dataId).values,v=r.data.get(a.dataId).values,w=nR.util.computeStrides(a.shape),C=nR.util.computeStrides(x.shape);for(let e=0;e<f;e++){let t=4*e,r=y[t],n=y[t+1],a=y[t+2],s=y[t+3],i=b[e];if(i>=h)continue;let o=m>1?(a-r)*(d-1)/(m-1):0,f=g>1?(s-n)*(p-1)/(g-1):0;for(let t=0;t<m;t++){let h=m>1?r*(d-1)+t*o:.5*(r+a)*(d-1);if(h<0||h>d-1){for(let r=0;r<g;r++)for(let n=0;n<c;n++){let a=n+r*C[2]+t*C[1]+e*C[0];x.values[a]=u}continue}if("bilinear"===l){let r=Math.floor(h),a=Math.ceil(h),o=h-r;for(let l=0;l<g;l++){let h=g>1?n*(p-1)+l*f:.5*(n+s)*(p-1);if(h<0||h>p-1){for(let r=0;r<c;r++){let n=r+l*C[2]+t*C[1]+e*C[0];x.values[n]=u}continue}let d=Math.floor(h),m=Math.ceil(h),y=h-d;for(let n=0;n<c;n++){let s=n+d*w[2]+r*w[1]+i*w[0],u=v[s],h=v[s=n+m*w[2]+r*w[1]+i*w[0]],p=v[s=n+d*w[2]+a*w[1]+i*w[0]],c=v[s=n+m*w[2]+a*w[1]+i*w[0]],f=u+(h-u)*y,g=p+(c-p)*y;s=n+l*C[2]+t*C[1]+e*C[0],x.values[s]=f+(g-f)*o}}}else for(let r=0;r<g;++r){let a=g>1?n*(p-1)+r*f:.5*(n+s)*(p-1);if(a<0||a>p-1){for(let n=0;n<c;n++){let a=n+r*C[2]+t*C[1]+e*C[0];x.values[a]=u}continue}let o=Math.round(a),l=Math.round(h);for(let n=0;n<c;n++){let a=n+o*w[2]+l*w[1]+i*w[0],s=n+r*C[2]+t*C[1]+e*C[0];x.values[s]=v[a]}}}}return r.makeTensorInfo(x.shape,x.dtype,x.values)}};var tN=tN;let fh={kernelName:x.Cumprod,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,exclusive:i,reverse:o}=n;h7(a,"cumprod");let l=tN.getAxesPermutation([s],a.shape.length),u=a;null!=l&&(u=pv({inputs:{x:a},backend:r,attrs:{perm:l}}));let h=tN.getInnerMostAxes(1,a.shape.length)[0];if(h!==u.shape.length-1)throw Error(`backend.cumprod in CPU expects an inner-most axis=${u.shape.length-1} but got axis=${h}`);let d=(0,py.upcastType)(u.dtype,"int32"),p=nR.util.makeOnesTypedArray(nR.util.sizeFromShape(u.shape),d),c=r.data.get(u.dataId).values,f=u.shape[u.shape.length-1],m=o?(e,t)=>e+f-t-1:(e,t)=>e+t;for(let e=0;e<c.length;e+=f)for(let t=0;t<f;t++){let r=m(e,t);if(0===t)p[r]=i?1:c[r];else{let n=m(e,t-1);p[r]=i?c[n]*p[n]:c[r]*p[n]}}let g=r.makeTensorInfo(u.shape,d,p);if(null!=l){let e=pv({inputs:{x:g},backend:r,attrs:{perm:tN.getUndoAxesPermutation(l)}});return r.disposeIntermediateTensorInfo(g),r.disposeIntermediateTensorInfo(u),e}return g}};var tN=tN;let fd={kernelName:x.Cumsum,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,exclusive:i,reverse:o}=n;h7(a,"cumsum");let l=tN.getAxesPermutation([s],a.shape.length),u=a;null!=l&&(u=pv({inputs:{x:a},backend:r,attrs:{perm:l}}));let h=tN.getInnerMostAxes(1,a.shape.length)[0];if(h!==u.shape.length-1)throw Error(`backend.cumsum in CPU expects an inner-most axis=${u.shape.length-1} but got axis=${h}`);let d=(0,py.upcastType)(u.dtype,"int32"),p=nR.util.makeZerosTypedArray(nR.util.sizeFromShape(u.shape),d),c=r.data.get(u.dataId).values,f=u.shape[u.shape.length-1],m=o?(e,t)=>e+f-t-1:(e,t)=>e+t;for(let e=0;e<c.length;e+=f)for(let t=0;t<f;t++){let r=m(e,t);if(0===t)p[r]=i?0:c[r];else{let n=m(e,t-1);p[r]=i?c[n]+p[n]:c[r]+p[n]}}let g=r.makeTensorInfo(u.shape,d,p);if(null!=l){let e=pv({inputs:{x:g},backend:r,attrs:{perm:tN.getUndoAxesPermutation(l)}});return r.disposeIntermediateTensorInfo(g),r.disposeIntermediateTensorInfo(u),e}return g}},fp={kernelName:x.DenseBincount,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,weights:s}=t,{size:i,binaryOutput:o}=n;if(1===a.shape.length){let e=dw(r.data.get(a.dataId).values,r.data.get(s.dataId).values,s.dtype,s.shape,i);return r.makeTensorInfo([i],s.dtype,e)}if(2===a.shape.length){let e=dC(r.bufferSync(a),r.bufferSync(s),i,o);return r.makeTensorInfo(e.shape,s.dtype,e.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${a.shape.length}.`)}},fc={kernelName:x.DepthToSpace,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{blockSize:s,dataFormat:i}=n;nR.util.assert("NHWC"===i,()=>`Only NHWC dataFormat supported on CPU for depthToSpace. Got ${i}`);let o=a.shape[0],l=a.shape[1],u=a.shape[2],h=a.shape[3],d=l*s,p=u*s,c=h/(s*s),f=r.data.get(a.dataId).values,m=new Float32Array(o*d*p*c),g=0;for(let e=0;e<o;++e)for(let t=0;t<d;++t){let r=Math.floor(t/s),n=t%s;for(let t=0;t<p;++t){let a=Math.floor(t/s),i=t%s,o=(n*s+i)*c;for(let t=0;t<c;++t){let n=t+o+h*(a+u*(r+l*e));m[g++]=f[n]}}}return r.makeTensorInfo([o,d,p,c],a.dtype,m)}};var tN=tN;function ff(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,filter:s}=t,{strides:i,pad:o,dilations:l,dimRoundingMode:u}=n;h7([a,s],"depthwiseConv2DNative");let h=nR.util.computeStrides(a.shape),d=nR.util.computeStrides(s.shape),p=l;null==p&&(p=[1,1]),nR.util.assert(tN.eitherStridesOrDilationsAreOne(i,p),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${i} and dilations '${p}'`);let c=tN.computeConv2DInfo(a.shape,s.shape,i,p,o,u,!0),{filterHeight:f,filterWidth:m,dilationHeight:g,dilationWidth:x,padInfo:y}=c,b=y.left,v=y.top,w=c.outChannels/c.inChannels,C=new rm.TensorBuffer(c.outShape,a.dtype),I=r.data.get(a.dataId).values,k=r.data.get(s.dataId).values,S=C.values;for(let e=0;e<c.batchSize;++e){let t=e*h[0],r=e*C.strides[0];for(let e=0;e<c.outHeight;++e){let n=r+e*C.strides[1],a=e*c.strideHeight-v;for(let e=0;e<f;++e){let r=a+e*g;if(r<0||r>=c.inHeight)continue;let s=e*d[0],i=t+r*h[1];for(let e=0;e<c.outWidth;++e){let t=n+e*C.strides[2],r=e*c.strideWidth-b;for(let e=0;e<m;++e){let n=r+e*x;if(n<0||n>=c.inWidth)continue;let a=s+e*d[1],o=i+n*c.inChannels,l=t,u=a;for(let e=0;e<c.inChannels;++e){let t=I[o+e];for(let e=0;e<w;++e)S[l+e]+=t*k[u+e];l+=w,u+=w}}}}}}return r.makeTensorInfo(C.shape,C.dtype,C.values)}let fm={kernelName:x.DepthwiseConv2dNative,backendName:"cpu",kernelFunc:ff};var tN=tN;let fg={kernelName:x.DepthwiseConv2dNativeBackpropFilter,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,dy:s}=t,{strides:i,dilations:o,pad:l,dimRoundingMode:u,filterShape:h}=n;h7([a,s],"depthwiseConv2dNativeBackpropFilter");let d=tN.computeConv2DInfo(a.shape,h,i,o,l,u,!0),{strideHeight:p,strideWidth:c,filterHeight:f,filterWidth:m}=d,g=new rm.TensorBuffer(d.filterShape,"float32"),x=d.padInfo.left,y=d.padInfo.top,b=d.outChannels/d.inChannels,v=r.data.get(a.dataId).values,w=new rm.TensorBuffer(a.shape,a.dtype,v),C=r.data.get(s.dataId).values,I=new rm.TensorBuffer(s.shape,s.dtype,C);for(let e=0;e<f;++e){let t=Math.max(0,Math.ceil((y-e)/p)),r=Math.min(d.outHeight,(d.inHeight+y-e)/p);for(let n=0;n<m;++n){let a=Math.max(0,Math.ceil((x-n)/c)),s=Math.min(d.outWidth,(d.inWidth+x-n)/c);for(let i=0;i<d.outChannels;++i){let o=Math.trunc(i/b),l=i%b,u=0;for(let l=0;l<d.batchSize;++l)for(let h=t;h<r;++h){let t=e+h*p-y;for(let e=a;e<s;++e){let r=n+e*c-x;u+=w.get(l,t,r,o)*I.get(l,h,e,i)}}g.set(u,e,n,o,l)}}}return r.makeTensorInfo(g.shape,g.dtype,g.values)}};var tN=tN;let fx={kernelName:x.DepthwiseConv2dNativeBackpropInput,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,filter:s}=t,{strides:i,dilations:o,pad:l,dimRoundingMode:u,inputShape:h}=n;h7([a,s],"depthwiseConv2DNativeBackpropInput");let d=nR.util.computeStrides(a.shape),p=nR.util.computeStrides(s.shape),c=tN.computeConv2DInfo(h,s.shape,i,o,l,u,!0),f=new rm.TensorBuffer(c.inShape,"float32"),m=f.values,[g,x,y]=f.strides,b=r.data.get(a.dataId).values,[v,w,C]=d,I=r.data.get(s.dataId).values,[k,S,T]=p,{batchSize:N,filterHeight:$,filterWidth:R,inChannels:A,inHeight:E,inWidth:F,outChannels:D,outHeight:O,outWidth:L,strideHeight:z,strideWidth:_}=c,M=$-1-c.padInfo.top,P=R-1-c.padInfo.left,B=D/A;for(let e=0;e<N;++e)for(let t=0;t<A;++t)for(let r=0;r<E;++r){let n=r-M,a=Math.max(0,Math.ceil(n/z)),s=Math.min(O,($+n)/z);for(let i=0;i<F;++i){let o=i-P,l=Math.max(0,Math.ceil(o/_)),u=Math.min(L,(R+o)/_),h=0;for(let r=a;r<s;++r){let a=r*z-n;for(let n=l;n<u;++n){let s=n*_-o,i=v*e+w*r+C*n,l=k*($-1-a)+S*(R-1-s)+T*t;for(let e=0;e<B;++e)h+=b[i+(t*B+e)]*I[l+e]}}m[g*e+x*r+y*i+t]=h}}return r.makeTensorInfo(f.shape,f.dtype,f.values)}},fy={kernelName:x.Diag,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{x:n}=t,a=nR.util.sizeFromShape(n.shape),s=r.data.get(n.dataId).values,i=(0,h5.buffer)([a,a],n.dtype),o=i.values;for(let e=0;e<s.length;e++)o[e*a+e]=s[e];let l=[...n.shape,...n.shape];return r.makeTensorInfo(l,i.dtype,i.values)}};var tN=tN;let fb={kernelName:x.Dilation2D,backendName:"cpu",kernelFunc:({inputs:e,backend:t,attrs:r})=>{let{x:n,filter:a}=e,{strides:s,pad:i,dilations:o}=r,l=t.data.get(n.dataId).values,u=n.shape.length,h=t.data.get(a.dataId).values,d=a.shape.length,{batchSize:p,inHeight:c,inWidth:f,inChannels:m,outHeight:g,outWidth:x,padInfo:y,strideHeight:b,strideWidth:v,filterHeight:w,filterWidth:C,dilationHeight:I,dilationWidth:k,outShape:S}=tN.computeDilation2DInfo(n.shape,a.shape,s,i,"NHWC",o),T=nR.util.sizeFromShape(S),N=S.length,$=nR.util.getArrayFromDType(n.dtype,T);for(let e=0;e<p;++e)for(let t=0;t<g;++t){let r=t*b-y.top;for(let s=0;s<x;++s){let i=s*v-y.left;for(let o=0;o<m;++o){let p=Number.MIN_SAFE_INTEGER;for(let t=0;t<w;++t){let s=r+t*I;if(s>=0&&s<c)for(let r=0;r<C;++r){let c=i+r*k;if(c>=0&&c<f){let i=nR.util.locToIndex([e,s,c,o],u,nR.util.computeStrides(n.shape)),f=nR.util.locToIndex([t,r,o],d,nR.util.computeStrides(a.shape)),m=l[i]+h[f];m>p&&(p=m)}}}$[nR.util.locToIndex([e,t,s,o],N,nR.util.computeStrides(S))]=p}}}return{dataId:t.write(nR.util.toTypedArray($,n.dtype),S,n.dtype),shape:S,dtype:n.dtype}}};var tN=tN;let fv={kernelName:x.Dilation2DBackpropFilter,backendName:"cpu",kernelFunc:({inputs:e,backend:t,attrs:r})=>{let{x:n,filter:a,dy:s}=e,{strides:i,pad:o,dilations:l}=r,u=nR.util.toNestedArray(n.shape,t.data.get(n.dataId).values),h=nR.util.toNestedArray(a.shape,t.data.get(a.dataId).values),{batchSize:d,inHeight:p,inWidth:c,inChannels:f,outHeight:m,outWidth:g,padInfo:y,strideHeight:b,strideWidth:v,filterHeight:w,filterWidth:C,dilationHeight:I,dilationWidth:k,outShape:S}=tN.computeDilation2DInfo(n.shape,a.shape,i,o,"NHWC",l);nR.util.assert(s.rank===S.length,()=>`Error in ${x.Dilation2DBackpropFilter}, dy must have the same rank as output ${S.length}, but got ${s.rank}`);let T=nR.util.toNestedArray(S,t.data.get(s.dataId).values),N=nR.util.makeZerosNestedTypedArray(a.shape,a.dtype);for(let e=0;e<d;++e)for(let t=0;t<m;++t){let r=t*b-y.top;for(let n=0;n<g;++n){let a=n*v-y.left;for(let s=0;s<f;++s){let i=Number.MIN_SAFE_INTEGER,o=0,l=0;for(let t=0;t<w;++t){let n=r+t*I;if(n>=0&&n<p)for(let r=0;r<C;++r){let d=a+r*k;if(d>=0&&d<c){let a=u[e][n][d][s]+h[t][r][s];a>i&&(i=a,o=t,l=r)}}}N[o][l][s]+=T[e][t][n][s]}}}return{dataId:t.write(nR.util.toTypedArray(N,n.dtype),a.shape,a.dtype),shape:a.shape,dtype:a.dtype}}};var tN=tN;let fw={kernelName:x.Dilation2DBackpropInput,backendName:"cpu",kernelFunc:({inputs:e,backend:t,attrs:r})=>{let{x:n,filter:a,dy:s}=e,{strides:i,pad:o,dilations:l}=r,u=nR.util.toNestedArray(n.shape,t.data.get(n.dataId).values),h=nR.util.toNestedArray(a.shape,t.data.get(a.dataId).values),{batchSize:d,inHeight:p,inWidth:c,inChannels:f,outHeight:m,outWidth:g,padInfo:y,strideHeight:b,strideWidth:v,filterHeight:w,filterWidth:C,dilationHeight:I,dilationWidth:k,outShape:S}=tN.computeDilation2DInfo(n.shape,a.shape,i,o,"NHWC",l);nR.util.assert(s.rank===S.length,()=>`Error in ${x.Dilation2DBackpropInput}, dy must have the same rank as output ${S.length}, but got ${s.rank}`);let T=nR.util.toNestedArray(S,t.data.get(s.dataId).values),N=nR.util.makeZerosNestedTypedArray(n.shape,n.dtype);for(let e=0;e<d;++e)for(let t=0;t<m;++t){let r=t*b-y.top;for(let n=0;n<g;++n){let a=n*v-y.left;for(let s=0;s<f;++s){let i=Number.MIN_SAFE_INTEGER,o=r<0?0:r,l=a<0?0:a;for(let t=0;t<w;++t){let n=r+t*I;if(n>=0&&n<p)for(let r=0;r<C;++r){let d=a+r*k;if(d>=0&&d<c){let a=u[e][n][d][s]+h[t][r][s];a>i&&(i=a,o=n,l=d)}}}N[e][o][l][s]+=T[e][t][n][s]}}}return{dataId:t.write(nR.util.toTypedArray(N,n.dtype),n.shape,n.dtype),shape:n.shape,dtype:n.dtype}}},fC={kernelName:x.Draw,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{image:a}=t,{canvas:s,options:i}=n,{contextOptions:o,imageOptions:l}=i||{},u=(null==l?void 0:l.alpha)||1,h=(null==o?void 0:o.contextType)||"2d";if("2d"!==h)throw Error(`Context type ${o.contextType} is not supported by the CPU backend.`);let d=s.getContext(h,(null==o?void 0:o.contextAttributes)||{});if(null==d)throw Error(`Could not get the context with ${h} type.`);let[p,c]=a.shape.slice(0,2),f=2===a.shape.length?1:a.shape[2],m=r.data.get(a.dataId).values,g="float32"===a.dtype?255:1,x=new Uint8ClampedArray(c*p*4);for(let e=0;e<p*c;++e){let t=[0,0,0,255*u];for(let r=0;r<f;r++){let n=m[e*f+r];if("float32"===a.dtype){if(n<0||n>1)throw Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${n}.`)}else if("int32"===a.dtype&&(n<0||n>255))throw Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${n}.`);1===f?(t[0]=n*g,t[1]=n*g,t[2]=n*g):t[r]=n*g}let r=4*e;x[r+0]=Math.round(t[0]),x[r+1]=Math.round(t[1]),x[r+2]=Math.round(t[2]),x[r+3]=Math.round(t[3])}s.width=c,s.height=p;let y=new ImageData(x,c,p);return d.putImageData(y,0,0),a}};var tN=tN,tN=tN;function fI(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{axis:i,keepDims:o}=a;h7(s,"sum");let l=(t="bool"===s.dtype?dc({inputs:{x:s},backend:n,attrs:{dtype:"int32"}}):dl({inputs:{x:s},backend:n})).shape.length,u=nR.util.parseAxisParam(i,t.shape),h=tN.getAxesPermutation(u,l),d=u,p=t;null!=h&&(p=pv({inputs:{x:t},backend:n,attrs:{perm:h}}),d=tN.getInnerMostAxes(d.length,l)),tN.assertAxesAreInnerMostDims("sum",d,p.shape.length);let[c,f]=tN.computeOutAndReduceShapes(p.shape,d),m=di(n,c,tN.upcastType(p.dtype,"int32")),g=nR.util.sizeFromShape(f),x=n.data.get(m.dataId).values,y=n.data.get(p.dataId).values;for(let e=0;e<x.length;++e){let t=e*g,r=0;for(let e=0;e<g;++e)r+=y[t+e];x[e]=r}if(o){let e=tN.expandShapeToKeepDim(m.shape,u),t=m;m=cb({inputs:{x:m},backend:n,attrs:{shape:e}}),n.disposeIntermediateTensorInfo(t)}return n.disposeIntermediateTensorInfo(t),null!=h&&n.disposeIntermediateTensorInfo(p),m}let fk={kernelName:x.Sum,backendName:"cpu",kernelFunc:fI},fS={kernelName:x.Einsum,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{equation:a}=n,{allDims:s,summedDims:i,idDims:o}=tN.decodeEinsumEquation(a,t.length);tN.checkEinsumDimSizes(s.length,o,t);let{path:l,steps:u}=tN.getEinsumComputePath(i,o),h=u.length,d=null,p=s.length,c=[];for(let e=0;e<h;++e){for(let n of u[e]){let e,{permutationIndices:a,expandDims:s}=tN.getEinsumPermutation(p,o[n]);tN.isIdentityPermutation(a)?e=t[n]:(e=pv({inputs:{x:t[n]},backend:r,attrs:{perm:a}}),c.push(e));let i=e.shape.slice();for(let e=0;e<s.length;++e)i.splice(s[e],0,1);nR.util.arraysEqual(e.shape,i)||(e=cb({inputs:{x:e},backend:r,attrs:{shape:i}}),c.push(e)),null===d?d=e:(d=pd({inputs:{a:e,b:d},backend:r}),c.push(d))}e<h-1&&(l[e]>=0&&(d=fI({inputs:{x:d},backend:r,attrs:{axis:l[e]-(s.length-p),keepDims:!1}}),c.push(d)),p--)}for(let e of c)e!==d&&r.disposeIntermediateTensorInfo(e);return d}},fT={kernelName:x.EluGrad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{dy:n,y:a}=t;h7([n,a],"eluGrad");let s=new Float32Array(nR.util.sizeFromShape(a.shape)),i=r.data.get(a.dataId).values,o=r.data.get(n.dataId).values;for(let e=0;e<i.length;++e){let t=i[e];t>=0?s[e]=o[e]:s[e]=o[e]*(t+1)}return r.makeTensorInfo(a.shape,"float32",s)}};var tN=tN;let fN=tN.ERF_P,f$=tN.ERF_A1,fR=tN.ERF_A2,fA=tN.ERF_A3,fE=tN.ERF_A4,fF=tN.ERF_A5,fD=dN(x.Erf,e=>{let t=Math.sign(e),r=Math.abs(e),n=1/(1+fN*r);return t*(1-((((fF*n+fE)*n+fA)*n+fR)*n+f$)*n*Math.exp(-r*r))}),fO={kernelName:x.Erf,backendName:"cpu",kernelFunc:fD};function fL(e){let{inputs:t,backend:r,attrs:n}=e,{input:a}=t,{dim:s}=n,i=a.shape.length,o=a.shape.slice(),l=s;return s<0&&(nR.util.assert(-(i+1)<=s,()=>`Axis must be in the interval [${-(i+1)}, ${i}]`),l=i+s+1),o.splice(l,0,1),cb({inputs:{x:a},backend:r,attrs:{shape:o}})}let fz={kernelName:x.ExpandDims,backendName:"cpu",kernelFunc:fL};var tN=tN;let f_=dn((e,t)=>e/t),fM=dm(x.RealDiv,f_),fP={kernelName:x.RealDiv,backendName:"cpu",kernelFunc:fM};function fB(e,t,r){let n=e.shape,a=n[0],s=n[1],i=r.data.get(e.dataId),o=i.complexTensorInfos.real,l=i.complexTensorInfos.imag,u=[a,s],h=nR.util.sizeFromShape(u),d=nR.util.getTypedArrayFromDType("float32",h),p=nR.util.getTypedArrayFromDType("float32",h);for(let e=0;e<a;e++){let n=pG({inputs:{x:o},backend:r,attrs:{begin:[e,0],size:[1,s]}}),a=pG({inputs:{x:l},backend:r,attrs:{begin:[e,0],size:[1,s]}}),i=da({inputs:{real:n,imag:a},backend:r}),{real:u,imag:h}=function(e,t,r){var n;let a=nR.util.sizeFromShape(e.shape),s=r.data.get(e.dataId),i=r.data.get(s.complexTensorInfos.real.dataId).values,o=r.data.get(s.complexTensorInfos.imag.dataId).values;if(((n=a)&n-1)==0){let n=function e(t,r,n,a,s){if(1===n)return{real:t,imag:r};let i=tN.mergeRealAndImagArrays(t,r),o=n/2,l=tN.complexWithEvenIndex(i),u=l.real,h=l.imag,d=[u.length],p=s.makeTensorInfo(d,"float32",u),c=s.makeTensorInfo(d,"float32",h),f=da({inputs:{real:p,imag:c},backend:s}),m=tN.complexWithOddIndex(i),g=m.real,x=m.imag,y=[g.length],b=s.makeTensorInfo(y,"float32",g),v=s.makeTensorInfo(y,"float32",x),w=da({inputs:{real:b,imag:v},backend:s}),C=e(u,h,o,a,s),I=C.real,k=C.imag,S=[I.length],T=s.makeTensorInfo(S,"float32",I),N=s.makeTensorInfo(S,"float32",k),$=da({inputs:{real:T,imag:N},backend:s}),R=e(g,x,o,a,s),A=R.real,E=R.imag,F=[A.length],D=s.makeTensorInfo(F,"float32",A),O=s.makeTensorInfo(F,"float32",E),L=da({inputs:{real:D,imag:O},backend:s}),z=tN.exponents(n,a),_=[z.real.length],M=s.makeTensorInfo(_,"float32",z.real),P=s.makeTensorInfo(_,"float32",z.imag),B=da({inputs:{real:M,imag:P},backend:s}),W=pd({inputs:{a:B,b:L},backend:s}),G=db({inputs:{a:$,b:W},backend:s}),U=p9({inputs:{a:$,b:W},backend:s}),V=dh({inputs:{input:G},backend:s}),H=dh({inputs:{input:U},backend:s}),q=c4({inputs:{input:G},backend:s}),j=c4({inputs:{input:U},backend:s}),X=c6({inputs:[V,H],backend:s,attrs:{axis:0}}),K=c6({inputs:[q,j],backend:s,attrs:{axis:0}}),Y=s.data.get(X.dataId).values,Z=s.data.get(K.dataId).values;return s.disposeIntermediateTensorInfo(p),s.disposeIntermediateTensorInfo(c),s.disposeIntermediateTensorInfo(f),s.disposeIntermediateTensorInfo(b),s.disposeIntermediateTensorInfo(v),s.disposeIntermediateTensorInfo(w),s.disposeIntermediateTensorInfo(T),s.disposeIntermediateTensorInfo(N),s.disposeIntermediateTensorInfo($),s.disposeIntermediateTensorInfo(D),s.disposeIntermediateTensorInfo(O),s.disposeIntermediateTensorInfo(L),s.disposeIntermediateTensorInfo(M),s.disposeIntermediateTensorInfo(P),s.disposeIntermediateTensorInfo(B),s.disposeIntermediateTensorInfo(W),s.disposeIntermediateTensorInfo(G),s.disposeIntermediateTensorInfo(U),s.disposeIntermediateTensorInfo(V),s.disposeIntermediateTensorInfo(q),s.disposeIntermediateTensorInfo(H),s.disposeIntermediateTensorInfo(j),s.disposeIntermediateTensorInfo(X),s.disposeIntermediateTensorInfo(K),{real:Y,imag:Z}}(i,o,a,t,r),s=[e.shape[0],e.shape[1]];if(t){let e=r.makeTensorInfo(s,"float32",n.real),t=r.makeTensorInfo(s,"float32",n.imag),i=r.makeTensorInfo([],"float32",nR.util.createScalarValue(a,"float32")),o=dl({inputs:{x:i},backend:r}),l=fP.kernelFunc({inputs:{a:e,b:i},backend:r}),u=fP.kernelFunc({inputs:{a:t,b:o},backend:r}),h=r.data.get(l.dataId).values,d=r.data.get(u.dataId).values;return r.disposeIntermediateTensorInfo(e),r.disposeIntermediateTensorInfo(t),r.disposeIntermediateTensorInfo(i),r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(l),r.disposeIntermediateTensorInfo(u),{real:h,imag:d}}return n}{let e=function(e,t,r){let n=new Float32Array(2*t);for(let a=0;a<t;a++){let s=0,i=0;for(let n=0;n<t;n++){let o=tN.exponent(a*n,t,r),l=tN.getComplexWithIndex(e,n);s+=l.real*o.real-l.imag*o.imag,i+=l.real*o.imag+l.imag*o.real}r&&(s/=t,i/=t),tN.assignToTypedArray(n,s,i,a)}return n}(tN.mergeRealAndImagArrays(i,o),a,t);return tN.splitRealAndImagArrays(e)}}(i,t,r),c=tN.mergeRealAndImagArrays(u,h);for(let t=0;t<s;t++){let r=tN.getComplexWithIndex(c,t);d[e*s+t]=r.real,p[e*s+t]=r.imag}r.disposeIntermediateTensorInfo(n),r.disposeIntermediateTensorInfo(a),r.disposeIntermediateTensorInfo(i)}let c=r.makeTensorInfo(u,"float32",d),f=r.makeTensorInfo(u,"float32",p),m=da({inputs:{real:c,imag:f},backend:r});return r.disposeIntermediateTensorInfo(c),r.disposeIntermediateTensorInfo(f),m}let fW={kernelName:x.FFT,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{input:n}=t,a=nR.util.sizeFromShape(n.shape),s=n.shape[n.shape.length-1],i=cb({inputs:{x:n},backend:r,attrs:{shape:[a/s,s]}}),o=fB(i,!1,r),l=cb({inputs:{x:o},backend:r,attrs:{shape:n.shape}});return r.disposeIntermediateTensorInfo(i),r.disposeIntermediateTensorInfo(o),l}};function fG(e){var t,r;let{backend:n,attrs:a}=e,{shape:s,value:i,dtype:o}=a,l=o||nR.util.inferDtype(i),u=nR.util.getArrayFromDType(l,nR.util.sizeFromShape(s));return t=u,r=i,t.fill(r),n.makeTensorInfo(s,l,u)}let fU={kernelName:x.Fill,backendName:"cpu",kernelFunc:fG},fV={kernelName:x.FlipLeftRight,backendName:"cpu",kernelFunc:({inputs:e,attrs:t,backend:r})=>{let{image:n}=e,a=nR.util.getTypedArrayFromDType(n.dtype,nR.util.sizeFromShape(n.shape)),[s,i,o,l]=n.shape,u=r.data.get(n.dataId).values;for(let e=0;e<s;e++){let t=e*o*i*l;for(let e=0;e<i;e++){let r=o*l*e;for(let e=0;e<o;e++){let n=e*l;for(let s=0;s<l;s++){let i=Math.round(o-e-1),h=t+r+n+s,d=u[h];i>=0&&i<o&&(d=u[t+r+i*l+s]),a[h]=d}}}}return{dataId:r.write(a,n.shape,n.dtype),shape:n.shape,dtype:n.dtype}}},fH={kernelName:x.FusedConv2D,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,filter:s,bias:i,preluActivationWeights:o}=t,{strides:l,pad:u,dataFormat:h,dilations:d,dimRoundingMode:p,activation:c,leakyreluAlpha:f}=n,m=c7({inputs:{x:a,filter:s},backend:r,attrs:{strides:l,pad:u,dataFormat:h,dilations:d,dimRoundingMode:p}});if(i){let e=m;if("NCHW"===h&&1===i.shape.length&&1!==i.shape[0]){let e=cb({inputs:{x:i},backend:r,attrs:{shape:[i.shape[0],1,1]}});m=db({inputs:{a:m,b:e},backend:r}),r.disposeIntermediateTensorInfo(e)}else m=db({inputs:{a:m,b:i},backend:r});r.disposeIntermediateTensorInfo(e)}if(c){let e=m;if("NCHW"===h&&"prelu"===c&&1===o.shape.length&&1!==o.shape[0]){let e=cb({inputs:{x:o},backend:r,attrs:{shape:[o.shape[0],1,1]}});m=cx(r,m,c,e,f),r.disposeIntermediateTensorInfo(e)}else m=cx(r,m,c,o,f);r.disposeIntermediateTensorInfo(e)}return m}},fq={kernelName:x.FusedDepthwiseConv2D,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,filter:s,bias:i,preluActivationWeights:o}=t,{strides:l,pad:u,dataFormat:h,dilations:d,dimRoundingMode:p,activation:c,leakyreluAlpha:f}=n,m=ff({inputs:{x:a,filter:s},backend:r,attrs:{strides:l,pad:u,dataFormat:h,dilations:d,dimRoundingMode:p}});if(i){let e=m;m=db({inputs:{a:m,b:i},backend:r}),r.disposeIntermediateTensorInfo(e)}if(c){let e=m;m=cx(r,m,c,o,f),r.disposeIntermediateTensorInfo(e)}return m}};var tN=tN;let fj={kernelName:x.GatherNd,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{params:n,indices:a}=t,s=nR.util.sizeFromShape(n.shape),i=a.shape,o=i[i.length-1],[l,u,h,d]=tN.prepareAndValidate(n,a);if(0===u)return r.makeTensorInfo(l,n.dtype,[]);let p=dX(r.data.get(a.dataId).values,r.bufferSync(n),n.dtype,u,o,h,d,n.shape,s);return r.makeTensorInfo(l,n.dtype,p.values)}};var tN=tN;let fX={kernelName:x.GatherV2,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,indices:s}=t,{axis:i,batchDims:o}=n;h7([a,s],"gatherV2");let l=nR.util.parseAxisParam(i,a.shape)[0],u=r.data.get(s.dataId).values,h=a.shape[l];for(let e=0;e<u.length;++e){let t=u[e];nR.util.assert(t<=h-1&&t>=0,()=>`GatherV2: the index value ${t} is not in [0, ${h-1}]`)}let d=o;null==o&&(d=0);let p=nR.util.sizeFromShape(s.shape),c=tN.segment_util.collectGatherOpShapeInfo(a,s,l,d),f=cb({inputs:{x:a},backend:r,attrs:{shape:[c.batchSize,c.outerSize,c.dimSize,c.sliceSize]}}),m=cb({inputs:{x:s},backend:r,attrs:{shape:[c.batchSize,p/c.batchSize]}}),g=[c.batchSize,c.outerSize,p/c.batchSize,c.sliceSize],x=r.bufferSync(m),y=dK(r.bufferSync(f),x,g);return r.disposeIntermediateTensorInfo(f),r.disposeIntermediateTensorInfo(m),r.makeTensorInfo(c.outputShape,y.dtype,y.values)}},fK={kernelName:x.IFFT,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{input:n}=t,a=nR.util.sizeFromShape(n.shape),s=n.shape[n.shape.length-1],i=cb({inputs:{x:n},backend:r,attrs:{shape:[a/s,s]}}),o=fB(i,!0,r),l=cb({inputs:{x:o},backend:r,attrs:{shape:n.shape}});return r.disposeIntermediateTensorInfo(i),r.disposeIntermediateTensorInfo(o),l}},fY=dN(x.IsFinite,e=>+!!Number.isFinite(e),"bool"),fZ={kernelName:x.IsFinite,backendName:"cpu",kernelFunc:fY},fJ=dN(x.IsInf,e=>+(Math.abs(e)===1/0),"bool"),fQ={kernelName:x.IsInf,backendName:"cpu",kernelFunc:fJ},f0=dN(x.IsNan,e=>+!!Number.isNaN(e),"bool"),f1={kernelName:x.IsNan,backendName:"cpu",kernelFunc:f0},f2={kernelName:x.LinSpace,backendName:"cpu",kernelFunc:function(e){let{backend:t,attrs:r}=e,{start:n,stop:a,num:s}=r,i=d7(n,a,s);return t.makeTensorInfo([i.length],"float32",i)}},f3=dN(x.Log1p,e=>Math.log1p(e)),f4={kernelName:x.Log1p,backendName:"cpu",kernelFunc:f3},f5=dn((e,t)=>e&&t),f6=dm(x.LogicalAnd,f5,null,"bool"),f8={kernelName:x.LogicalAnd,backendName:"cpu",kernelFunc:f6},f7=dN(x.LogicalNot,e=>+!e,"bool"),f9={kernelName:x.LogicalNot,backendName:"cpu",kernelFunc:f7},me=dn((e,t)=>e||t),mt=dm(x.LogicalOr,me,null,"bool"),mr={kernelName:x.LogicalOr,backendName:"cpu",kernelFunc:mt},mn={kernelName:x.LRN,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{depthRadius:s,bias:i,alpha:o,beta:l}=n;h7(a,"LRN");let u=a.shape[3],h=u-1,d=r.data.get(a.dataId).values,p=nR.util.sizeFromShape(a.shape),c=new Float32Array(p);for(let e=0;e<p;e++){let t=function(e){let t=e%u,r=e-t+Math.max(0,t-s),n=e-t+Math.min(t+s,h),a=0;for(;r<=n;r++){let e=d[r];a+=e*e}return a}(e),r=d[e]*Math.pow(i+o*t,-l);c[e]=r}return r.makeTensorInfo(a.shape,a.dtype,c)}},ma={kernelName:x.LRNGrad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,y:s,dy:i}=t,{depthRadius:o,bias:l,alpha:u,beta:h}=n;h7(i,"LRNGrad");let d=nR.util.sizeFromShape(i.shape),p=i.shape[3],c=r.data.get(i.dataId).values,f=r.data.get(a.dataId).values,m=r.data.get(s.dataId).values,g=new Float32Array(d);for(let e=0;e<d;e++){let t=e%p,r=e-t+Math.max(0,t-o),n=e-t+Math.min(p,t+o+1),a=0;for(let e=r;e<n;e++)a+=Math.pow(f[e],2);a=u*a+l;for(let t=r;t<n;t++){let r=-2*u*h*f[t]*m[e]/a;e===t&&(r+=Math.pow(a,-h)),r*=c[e],g[t]+=r}}return r.makeTensorInfo(i.shape,a.dtype,g)}};var tN=tN;function ms(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{reductionIndices:s,keepDims:i}=n,o=a.shape,l=o.length,u=nR.util.parseAxisParam(s,o),h=u,d=tN.getAxesPermutation(h,l),p=r.data.get(a.dataId).values;if(null!=d){let e=Array(l);for(let t=0;t<e.length;t++)e[t]=o[d[t]];p=pb(p,o,a.dtype,d,e),h=tN.getInnerMostAxes(h.length,l),o=e}h7(a,"max"),tN.assertAxesAreInnerMostDims("max",h,l);let[c,f]=tN.computeOutAndReduceShapes(o,h),m=pr(p,nR.util.sizeFromShape(f),c,a.dtype),g=r.write(m,c,a.dtype),x=c;return i&&(x=tN.expandShapeToKeepDim(c,u)),{dataId:g,shape:x,dtype:a.dtype}}let mi={kernelName:x.Max,backendName:"cpu",kernelFunc:ms};var tN=tN;let mo={kernelName:x.MaxPool,backendName:"cpu",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r;h7(s,"maxPool");let{filterSize:i,strides:o,pad:l,dimRoundingMode:u}=a;nR.util.assert(tN.eitherStridesOrDilationsAreOne(o,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '1'`);let h=tN.computePool2DInfo(s.shape,i,o,1,l,u);if(1===h.filterWidth&&1===h.filterHeight&&nR.util.arraysEqual(h.inShape,h.outShape))t=dl({inputs:{x:s},backend:n});else{let e=n.data.get(s.dataId).values,r=nR.util.computeStrides(s.shape),a=cV(e,s.shape,s.dtype,r,h,"max");t=n.makeTensorInfo(h.outShape,s.dtype,a.values)}return t}};var tN=tN;let ml={kernelName:x.MaxPool3D,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{filterSize:s,strides:i,pad:o,dimRoundingMode:l,dataFormat:u}=n;h7(a,"maxPool3d");let h=tN.computePool3DInfo(a.shape,s,i,1,o,l,u),d=cq(r.data.get(a.dataId).values,a.shape,a.dtype,nR.util.computeStrides(a.shape),h,"max");return r.makeTensorInfo(d.shape,"float32",d.values)}};var tN=tN;let mu={kernelName:x.MaxPool3DGrad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,input:s}=t,{filterSize:i,strides:o,pad:l,dimRoundingMode:u}=n;h7([a,s],"maxPool3DGrad");let h=tN.computePool3DInfo(s.shape,i,o,1,l,u),d=function(e,t){let r=(0,h5.buffer)(t.outShape,"int32"),n=t.strideDepth,a=t.strideHeight,s=t.strideWidth,i=t.dilationDepth,o=t.dilationHeight,l=t.dilationWidth,u=t.effectiveFilterDepth,h=t.effectiveFilterHeight,d=t.effectiveFilterWidth,p=t.padInfo.front,c=t.padInfo.top,f=t.padInfo.left;for(let m=0;m<t.batchSize;++m)for(let g=0;g<t.inChannels;++g)for(let x=0;x<t.outDepth;++x){let y=x*n-p,b=y;for(;b<0;)b+=i;let v=Math.min(t.inDepth,u+y);for(let n=0;n<t.outHeight;++n){let u=n*a-c,p=u;for(;p<0;)p+=o;let w=Math.min(t.inHeight,h+u);for(let a=0;a<t.outWidth;++a){let c=a*s-f,C=c;for(;C<0;)C+=l;let I=Math.min(t.inWidth,d+c),k=-1/0,S=-1;for(let t=b;t<v;t+=i){let r=t-y;for(let n=p;n<w;n+=o){let a=n-u;for(let s=C;s<I;s+=l){let i=s-c,o=e.get(m,t,n,s,g);o>=k&&(k=o,S=r*h*d+a*h+i)}}}r.set(S,m,x,n,a,g)}}}return r}(r.bufferSync(s),h),p=h.strideDepth,c=h.strideHeight,f=h.strideWidth,m=h.dilationDepth,g=h.dilationHeight,x=h.dilationWidth,y=h.effectiveFilterDepth,b=h.effectiveFilterHeight,v=h.effectiveFilterWidth,w=y-1-h.padInfo.front,C=v-1-h.padInfo.left,I=b-1-h.padInfo.top,k=(0,h5.buffer)(s.shape,"float32"),S=r.bufferSync(a);for(let e=0;e<h.batchSize;++e)for(let t=0;t<h.inChannels;++t)for(let r=0;r<h.inDepth;++r)for(let n=0;n<h.inHeight;++n)for(let a=0;a<h.inWidth;++a){let s=r-w,i=n-I,o=a-C,l=0;for(let r=0;r<y;r+=m){let n=(s+r)/p;if(!(n<0)&&!(n>=h.outDepth)&&Math.floor(n)===n)for(let a=0;a<b;a+=g){let s=(i+a)/c;if(!(s<0)&&!(s>=h.outHeight)&&Math.floor(s)===s)for(let i=0;i<v;i+=x){let u=(o+i)/f;if(u<0||u>=h.outWidth||Math.floor(u)!==u)continue;let p=+(y*b*v-1-d.get(e,n,s,u,t)===r*b*v+a*v+i);0!==p&&(l+=S.get(e,n,s,u,t)*p)}}}k.set(l,e,r,n,a,t)}return r.makeTensorInfo(k.shape,k.dtype,k.values)}};var tN=tN;let mh={kernelName:x.MaxPoolGrad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,input:s,output:i}=t;h7([s,i],"maxPoolGrad");let{filterSize:o,strides:l,pad:u,dimRoundingMode:h}=n,d=tN.computePool2DInfo(s.shape,o,l,1,u,h),p=r.data.get(s.dataId).values,c=(0,h5.buffer)(d.outShape,s.dtype,cH(p,s.shape,s.dtype,d).values),f=d.strideHeight,m=d.strideWidth,g=d.dilationHeight,x=d.dilationWidth,y=d.effectiveFilterHeight,b=d.effectiveFilterWidth,v=b-1-d.padInfo.left,w=y-1-d.padInfo.top,C=(0,h5.buffer)(s.shape,"float32"),I=r.data.get(a.dataId).values,k=(0,h5.buffer)(a.shape,"float32",I);for(let e=0;e<d.batchSize;++e)for(let t=0;t<d.inChannels;++t)for(let r=0;r<d.inHeight;++r)for(let n=0;n<d.inWidth;++n){let a=r-w,s=n-v,i=0;for(let r=0;r<y;r+=g){let n=(a+r)/f;if(!(n<0)&&!(n>=d.outHeight)&&Math.floor(n)===n)for(let a=0;a<b;a+=x){let o=(s+a)/m;if(o<0||o>=d.outWidth||Math.floor(o)!==o)continue;let l=+(y*b-1-c.get(e,n,o,t)===r*b+a);0!==l&&(i+=k.get(e,n,o,t)*l)}}C.set(i,e,r,n,t)}return r.makeTensorInfo(C.shape,C.dtype,C.values)}};var tN=tN;let md={kernelName:x.MaxPoolWithArgmax,backendName:"cpu",kernelFunc:({inputs:e,attrs:t,backend:r})=>{var n,a;let s,i,o,{x:l}=e,{filterSize:u,strides:h,pad:d,includeBatchInIndex:p}=t;h7(l,"MaxPoolWithArgmax");let c=r.data.get(l.dataId).values,f=tN.computePool2DInfo(l.shape,u,h,[1,1],d),[m,g]=(n=l.shape,a=l.dtype,s=nR.util.computeStrides(n),i=cV(c,n,a,s,f,"max"),o=cH(c,n,a,f,!0,p),[i.values,o.values]),x=r.write(m,f.outShape,l.dtype),y=r.write(g,f.outShape,l.dtype);return[{dataId:x,shape:f.outShape,dtype:l.dtype},{dataId:y,shape:f.outShape,dtype:"int32"}]}};var tN=tN;let mp={kernelName:x.Mean,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,keepDims:i}=n,o=nR.util.parseAxisParam(s,a.shape),l=tN.computeOutAndReduceShapes(a.shape,o)[1],u=nR.util.sizeFromShape(l),h=[],d=r.makeTensorInfo([],"float32",new Float32Array([u]));h.push(d);let p=dc({inputs:{x:a},backend:r,attrs:{dtype:"float32"}});h.push(p);let c=fM({inputs:{a:p,b:d},backend:r});h.push(c);let f=fI({inputs:{x:c},backend:r,attrs:{axis:s,keepDims:i}});return h.forEach(e=>r.disposeIntermediateTensorInfo(e)),f}};var tN=tN;let mc={kernelName:x.Min,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,keepDims:i}=n;h7(a,"min");let o=nR.util.parseAxisParam(s,a.shape),l=o,u=tN.getAxesPermutation(l,a.shape.length),h=a;null!=u&&(h=pv({inputs:{x:a},backend:r,attrs:{perm:u}}),l=tN.getInnerMostAxes(l.length,a.shape.length)),tN.assertAxesAreInnerMostDims("min",l,h.shape.length);let[d,p]=tN.computeOutAndReduceShapes(h.shape,l),c=nR.util.sizeFromShape(p),f=nR.util.makeZerosTypedArray(nR.util.sizeFromShape(d),h.dtype),m=r.data.get(h.dataId).values;for(let e=0;e<f.length;++e){let t=e*c,r=m[t];for(let e=0;e<c;++e){let n=m[t+e];(Number.isNaN(n)||n<r)&&(r=n)}f[e]=r}null!=u&&r.disposeIntermediateTensorInfo(h);let g=r.makeTensorInfo(d,h.dtype,f);if(i){let e=cb({inputs:{x:g},backend:r,attrs:{shape:tN.expandShapeToKeepDim(d,o)}});return r.disposeIntermediateTensorInfo(g),e}return g}},mf={kernelName:x.MirrorPad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{paddings:s,mode:i}=n;h7(a,"mirrorPad");let o=s.map((e,t)=>e[0]+a.shape[t]+e[1]),l=s.map(e=>e[0]),u=s.map((e,t)=>e[0]+a.shape[t]),h=+("reflect"!==i),d=r.data.get(a.dataId).values,p=a.shape.length,c=nR.util.computeStrides(a.shape),f=nR.util.sizeFromShape(o),m=o.length,g=nR.util.computeStrides(o),x=nR.util.getTypedArrayFromDType(a.dtype,f);for(let e=0;e<f;e++){let t=nR.util.indexToLoc(e,m,g);for(let e=0;e<m;e++)t[e]<l[e]?t[e]=2*l[e]-t[e]-h:t[e]>=u[e]&&(t[e]=(u[e]-1)*2-t[e]+h);t=t.map((e,t)=>e-l[t]);let r=nR.util.locToIndex(t,p,c);x[e]=d[r]}return{dataId:r.write(x,o,a.dtype),shape:o,dtype:a.dtype}}},mm=dn((e,t)=>{let r=e%t;return e<0&&t<0||e>=0&&t>=0?r:(r+t)%t}),mg=dm(x.Mod,mm),mx={kernelName:x.Mod,backendName:"cpu",kernelFunc:mg};var tN=tN;function my(e){let{inputs:t,backend:r,attrs:n}=e,{logits:a}=t,{dim:s}=n,i=a.shape.length,o=s;if(-1===o&&(o=i-1),o!==i-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${i} and dim was ${o}`);let l=nR.util.parseAxisParam([o],a.shape),u=ms({inputs:{x:a},backend:r,attrs:{reductionIndices:l,keepDims:!1}}),h=tN.expandShapeToKeepDim(u.shape,l),d=cb({inputs:{x:u},backend:r,attrs:{shape:h}}),p=p9({inputs:{a:a,b:d},backend:r}),c=d_({inputs:{x:p},backend:r}),f=fI({inputs:{x:c},backend:r,attrs:{axis:l,keepDims:!1}}),m=cb({inputs:{x:f},backend:r,attrs:{shape:h}}),g=fM({inputs:{a:c,b:m},backend:r});return r.disposeIntermediateTensorInfo(u),r.disposeIntermediateTensorInfo(d),r.disposeIntermediateTensorInfo(p),r.disposeIntermediateTensorInfo(c),r.disposeIntermediateTensorInfo(f),r.disposeIntermediateTensorInfo(m),g}let mb={kernelName:x.Softmax,backendName:"cpu",kernelFunc:my},mv={kernelName:x.Multinomial,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{logits:a}=t,{numSamples:s,seed:i,normalized:o}=n;h7(a,"multinomial");let l=o?a:my({inputs:{logits:a},backend:r,attrs:{dim:-1}}),u=l.shape[0],h=l.shape[1],d=r.data.get(l.dataId).values,p=[u,s],c=nR.util.makeZerosTypedArray(nR.util.sizeFromShape(p),"int32");for(let e=0;e<u;++e){let t=e*h,r=new Float32Array(h-1);r[0]=d[t];for(let e=1;e<r.length;++e)r[e]=r[e-1]+d[t+e];let n=ht.alea(i.toString()),a=e*s;for(let e=0;e<s;++e){let t=n();c[a+e]=r.length;for(let n=0;n<r.length;n++)if(t<r[n]){c[a+e]=n;break}}}return o||r.disposeIntermediateTensorInfo(l),r.makeTensorInfo(p,"int32",c)}};var h8=h8;let mw=h8.nonMaxSuppressionV3Impl,mC={kernelName:x.NonMaxSuppressionV3,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{boxes:a,scores:s}=t,{maxOutputSize:i,iouThreshold:o,scoreThreshold:l}=n;h7(a,"NonMaxSuppression");let{selectedIndices:u}=mw(r.data.get(a.dataId).values,r.data.get(s.dataId).values,i,o,l);return r.makeTensorInfo([u.length],"int32",new Int32Array(u))}};var h8=h8;let mI=h8.nonMaxSuppressionV4Impl,mk={kernelName:x.NonMaxSuppressionV4,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{boxes:a,scores:s}=t,{maxOutputSize:i,iouThreshold:o,scoreThreshold:l,padToMaxOutputSize:u}=n;h7(a,"NonMaxSuppressionPadded");let{selectedIndices:h,validOutputs:d}=mI(r.data.get(a.dataId).values,r.data.get(s.dataId).values,i,o,l,u);return[r.makeTensorInfo([h.length],"int32",new Int32Array(h)),r.makeTensorInfo([],"int32",new Int32Array([d]))]}};var h8=h8;let mS=h8.nonMaxSuppressionV5Impl,mT={kernelName:x.NonMaxSuppressionV5,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{boxes:a,scores:s}=t,{maxOutputSize:i,iouThreshold:o,scoreThreshold:l,softNmsSigma:u}=n;h7(a,"NonMaxSuppressionWithScore");let{selectedIndices:h,selectedScores:d}=mS(r.data.get(a.dataId).values,r.data.get(s.dataId).values,i,o,l,u);return[r.makeTensorInfo([h.length],"int32",new Int32Array(h)),r.makeTensorInfo([d.length],"float32",new Float32Array(d))]}},mN={kernelName:x.OneHot,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{indices:a}=t,{dtype:s,depth:i,onValue:o,offValue:l}=n;h7(a,"oneHot");let u=nR.util.sizeFromShape(a.shape),h=new Float32Array(u*i);h.fill(l);let d=r.data.get(a.dataId).values;for(let e=0;e<u;++e)d[e]>=0&&d[e]<i&&(h[e*i+d[e]]=o);return r.makeTensorInfo([...a.shape,i],s,h)}};function m$(e){let{inputs:t,backend:r}=e,{x:n}=t;if("string"===n.dtype)throw Error("zerosLike is not supported for string tensors");if("complex64"!==n.dtype)return fG({backend:r,attrs:{shape:n.shape,value:0,dtype:n.dtype}});{let e=dh({inputs:{input:n},backend:r}),t=m$({inputs:{x:e},backend:r}),a=c4({inputs:{input:n},backend:r}),s=m$({inputs:{x:a},backend:r}),i=da({inputs:{real:t,imag:s},backend:r});return r.disposeIntermediateTensorInfo(e),r.disposeIntermediateTensorInfo(t),r.disposeIntermediateTensorInfo(a),r.disposeIntermediateTensorInfo(s),i}}let mR={kernelName:x.ZerosLike,backendName:"cpu",kernelFunc:m$},mA={kernelName:x.OnesLike,backendName:"cpu",kernelFunc:function e(t){let{inputs:r,backend:n}=t,{x:a}=r;if("string"===a.dtype)throw Error("onesLike is not supported for string tensors");if("complex64"!==a.dtype)return fG({backend:n,attrs:{shape:a.shape,value:1,dtype:a.dtype}});{let t=dh({inputs:{input:a},backend:n}),r=e({inputs:{x:t},backend:n}),s=c4({inputs:{input:a},backend:n}),i=m$({inputs:{x:s},backend:n}),o=da({inputs:{real:r,imag:i},backend:n});return n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(r),n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(i),o}}};function mE(e){let{inputs:t,backend:r,attrs:n}=e,{axis:a}=n;if(1===t.length)return fL({inputs:{input:t[0]},backend:r,attrs:{dim:a}});let s=t[0].shape,i=t[0].dtype;t.forEach(e=>{nR.util.assertShapesMatch(s,e.shape,"All tensors passed to stack must have matching shapes"),nR.util.assert(i===e.dtype,()=>"All tensors passed to stack must have matching dtypes")});let o=[],l=c6({inputs:t.map(e=>{let t=fL({inputs:{input:e},backend:r,attrs:{dim:a}});return o.push(t),t}),backend:r,attrs:{axis:a}});return o.forEach(e=>r.disposeIntermediateTensorInfo(e)),l}let mF={kernelName:x.Pack,backendName:"cpu",kernelFunc:mE},mD={kernelName:x.PadV2,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{paddings:s,constantValue:i}=n;h7(a,"pad");let o=s.map((e,t)=>e[0]+a.shape[t]+e[1]),l=s.map(e=>e[0]),u=r.data.get(a.dataId).values,h=nR.util.sizeFromShape(a.shape),d=a.shape.length,p=nR.util.computeStrides(a.shape),c=nR.util.sizeFromShape(o),f=o.length,m=nR.util.computeStrides(o),g=nR.util.getTypedArrayFromDType(a.dtype,c);0!==i&&g.fill(i);for(let e=0;e<h;e++){let t=nR.util.indexToLoc(e,d,p).map((e,t)=>e+l[t]);g[nR.util.locToIndex(t,f,m)]=u[e]}return{dataId:r.write(g,o,a.dtype),shape:o,dtype:a.dtype}}},mO=dn((e,t)=>Math.pow(e,t)),mL=dm(x.Pow,mO),mz={kernelName:x.Pow,backendName:"cpu",kernelFunc:mL},m_={kernelName:x.RaggedGather,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{paramsNestedSplits:a,paramsDenseValues:s,indices:i}=t,{outputRaggedRank:o}=n,l=a.map(e=>r.data.get(e.dataId).values),u=a.map(e=>e.shape),h=r.data.get(s.dataId).values,d=r.data.get(i.dataId).values,[p,c,f]=pS(l,u,h,s.shape,s.dtype,d,i.shape,o),m=p.map(e=>r.makeTensorInfo([e.length],"int32",e)),g=r.makeTensorInfo(f,s.dtype,c);return m.concat([g])}},mM={kernelName:x.RaggedRange,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{starts:n,limits:a,deltas:s}=t,i=r.data.get(n.dataId).values,o=r.data.get(a.dataId).values,l=r.data.get(s.dataId).values,[u,h]=pT(i,n.shape,n.dtype,o,a.shape,l,s.shape);return[r.makeTensorInfo([u.length],"int32",u),r.makeTensorInfo([h.length],n.dtype,h)]}},mP={kernelName:x.RaggedTensorToTensor,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{shape:a,values:s,defaultValue:i,rowPartitionTensors:o}=t,{rowPartitionTypes:l}=n,u=r.data.get(a.dataId).values,h=r.data.get(s.dataId).values,d=r.data.get(i.dataId).values,p=o.map(e=>r.data.get(e.dataId).values),c=o.map(e=>e.shape),[f,m]=pE(u,a.shape,h,s.shape,s.dtype,d,i.shape,p,c,l);return r.makeTensorInfo(f,s.dtype,m)}},mB={kernelName:x.Range,backendName:"cpu",kernelFunc:function(e){let{backend:t,attrs:r}=e,{start:n,stop:a,dtype:s,step:i}=r,o=pF(n,a,i,s);return t.makeTensorInfo([o.length],s,o)}},mW=dN(x.Reciprocal,e=>1/e),mG={kernelName:x.Reciprocal,backendName:"cpu",kernelFunc:mW},mU={kernelName:x.ResizeBilinear,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{images:a}=t,{alignCorners:s,halfPixelCenters:i,size:o}=n;h7(a,"resizeBilinear");let l=nR.util.computeStrides(a.shape),[u,h]=o,[d,p,c,f]=a.shape,m=r.data.get(a.dataId).values,g=new Float32Array(nR.util.sizeFromShape([d,u,h,f])),x=[s&&u>1?p-1:p,s&&h>1?c-1:c],y=[s&&u>1?u-1:u,s&&h>1?h-1:h],b=0,v=x[0]/y[0],w=x[1]/y[1];for(let e=0;e<d;e++)for(let t=0;t<u;t++){let r,n=Math.max(0,Math.floor(r=i?v*(t+.5)-.5:v*t)),a=r-n,s=Math.min(p-1,Math.ceil(r)),o=e*l[0]+n*l[1],u=e*l[0]+s*l[1];for(let e=0;e<h;e++){let t,r=Math.max(0,Math.floor(t=i?w*(e+.5)-.5:w*e)),n=t-r,s=Math.min(c-1,Math.ceil(t)),h=o+r*l[2],d=u+r*l[2],p=o+s*l[2],x=u+s*l[2];for(let e=0;e<f;e++){let t=m[h+e],r=m[d+e],s=m[p+e],i=m[x+e],o=t+(s-t)*n,l=o+(r+(i-r)*n-o)*a;g[b++]=l}}}return r.makeTensorInfo([d,u,h,f],"float32",g)}},mV={kernelName:x.ResizeBilinearGrad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{images:a,dy:s}=t,{alignCorners:i}=n;h7([s,a],"resizeBilinearGrad");let o=nR.util.computeStrides(a.shape),[l,u,h,d]=a.shape,[,p,c]=s.shape,f=new Float32Array(l*u*h*d),m=[i&&p>1?u-1:u,i&&c>1?h-1:h],g=[i&&p>1?p-1:p,i&&c>1?c-1:c],x=m[0]/g[0],y=m[1]/g[1],b=r.data.get(s.dataId).values,v=0;for(let e=0;e<l;e++){let t=e*o[0];for(let e=0;e<p;e++){let r=e*x,n=Math.floor(r),a=Math.min(Math.ceil(r),u-1),s=t+n*o[1],i=t+a*o[1],l=r-n,p=1-l;for(let e=0;e<c;e++){let t=e*y,r=Math.floor(t),n=Math.min(Math.ceil(t),h-1),a=t-r,u=1-a,c=s+r*o[2],m=s+n*o[2],g=i+r*o[2],x=i+n*o[2],w=p*u,C=p*a,I=l*u,k=l*a;for(let e=0;e<d;e++){let t=b[v++];f[c+e]+=t*w,f[m+e]+=t*C,f[g+e]+=t*I,f[x+e]+=t*k}}}}return r.makeTensorInfo([l,h,u,d],"float32",f)}},mH={kernelName:x.ResizeNearestNeighbor,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{images:a}=t,{alignCorners:s,halfPixelCenters:i,size:o}=n;h7(a,"resizeNearestNeighbor");let l=nR.util.computeStrides(a.shape),[u,h]=o,[d,p,c,f]=a.shape,m=r.data.get(a.dataId).values,g=new Float32Array(d*u*h*f),x=[s&&u>1?p-1:p,s&&h>1?c-1:c],y=[s&&u>1?u-1:u,s&&h>1?h-1:h],b=x[0]/y[0],v=x[1]/y[1],w=0;for(let e=0;e<d;e++){let t=e*l[0];for(let e=0;e<u;e++){let r=i?b*(e+.5):b*e,n=Math.min(p-1,s?Math.round(r):Math.floor(r));i&&(n=Math.max(0,n));let a=t+n*l[1];for(let e=0;e<h;e++){let t=i?v*(e+.5):v*e,r=Math.min(c-1,s?Math.round(t):Math.floor(t));i&&(r=Math.max(0,r));let n=a+r*l[2];for(let e=0;e<f;e++){let t=m[n+e];g[w++]=t}}}}return r.makeTensorInfo([d,u,h,f],a.dtype,g)}},mq={kernelName:x.ResizeNearestNeighborGrad,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{images:a,dy:s}=t,{alignCorners:i}=n;h7([s,a],"resizeNearestNeighborGrad");let o=nR.util.computeStrides(a.shape),l=nR.util.computeStrides(s.shape),[u,h,d,p]=a.shape,[,c,f]=s.shape,m=new Float32Array(u*h*d*p),g=r.data.get(s.dataId).values,x=[i&&c>1?h-1:h,i&&f>1?d-1:d],y=[i&&c>1?c-1:c,i&&f>1?f-1:f],b=x[0]/y[0],v=x[1]/y[1],w=1/b,C=1/v,I=2*Math.ceil(w)+2,k=2*Math.ceil(C)+2;for(let e=0;e<u;e++){let t=e*o[0];for(let e=0;e<h;e++){let r=t+e*o[1],n=Math.floor(Math.floor(e*w)-I/2);for(let a=0;a<d;a++){let s=r+a*o[2],u=Math.floor(Math.floor(a*C)-k/2);for(let r=0;r<p;r++){let o=0;for(let s=0;s<I;s++){let p=s+n;if(p<0||p>=c)continue;let m=t+p*l[1],x=p*b;if(e===Math.min(h-1,i?Math.round(x):Math.floor(x)))for(let e=0;e<k;e++){let t=e+u;if(t<0||t>=f)continue;let n=m+t*l[2],s=t*v;a===Math.min(d-1,i?Math.round(s):Math.floor(s))&&(o+=g[n+r])}}m[s+r]=o}}}}return r.makeTensorInfo(a.shape,a.dtype,m)}},mj={kernelName:x.Reverse,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{dims:s}=n;h7(a,"reverse");let i=a.shape.length,o=nR.util.parseAxisParam(s,a.shape);if(0===i)return dl({inputs:{x:a},backend:r});let l=new rm.TensorBuffer(a.shape,a.dtype),u=r.bufferSync(a);for(let e=0;e<l.size;e++){let t=l.indexToLoc(e),r=t.slice();o.forEach(e=>r[e]=a.shape[e]-1-r[e]),l.set(u.get(...r),...t)}return r.makeTensorInfo(l.shape,l.dtype,l.values)}};var tN=tN;let mX={kernelName:x.RotateWithOffset,backendName:"cpu",kernelFunc:({inputs:e,attrs:t,backend:r})=>{let{image:n}=e,{radians:a,fillValue:s,center:i}=t,o=nR.util.getTypedArrayFromDType(n.dtype,nR.util.sizeFromShape(n.shape)),[l,u,h,d]=n.shape,[p,c]=tN.getImageCenter(i,u,h),f=Math.sin(a),m=Math.cos(a),g=r.data.get(n.dataId).values;for(let e=0;e<l;e++){let t=e*h*u*d;for(let e=0;e<u;e++){let r=h*d*e;for(let n=0;n<h;n++){let a=n*d;for(let i=0;i<d;i++){let x=[l,e,n,i],y=x[2],b=x[1],v=(y-p)*m-(b-c)*f,w=(y-p)*f+(b-c)*m;v=Math.round(v+p),w=Math.round(w+c);let C=s;"number"!=typeof s&&(C=3===i?255:s[i]),v>=0&&v<h&&w>=0&&w<u&&(C=g[t+h*d*w+v*d+i]),o[t+r+a+i]=C}}}}return{dataId:r.write(o,n.shape,n.dtype),shape:n.shape,dtype:n.dtype}}},mK=dN(x.Round,e=>{let t=Math.floor(e);return e-t<.5?Math.floor(e):e-t>.5?Math.ceil(e):t%2==0?t:t+1}),mY={kernelName:x.Round,backendName:"cpu",kernelFunc:mK};var tN=tN;let mZ={kernelName:x.ScatterNd,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{indices:a,updates:s}=t,{shape:i}=n,{sliceRank:o,numUpdates:l,sliceSize:u,strides:h,outputSize:d}=tN.calculateShapes(s,a,i),p=pz(r.bufferSync(a),r.bufferSync(s),i,d,u,l,o,h,0,!0);return r.makeTensorInfo(i,p.dtype,p.values)}},mJ={kernelName:x.SearchSorted,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{sortedSequence:a,values:s}=t,{side:i}=n,o=function(e,t,r,n,a,s){let i=nR.util.getArrayFromDType("int32",r*a);for(let o=0;o<r;++o){let r=e.slice(o*n,(o+1)*n),l=o*a;for(let e=0;e<a;++e)i[l+e]="left"===s?function(e,t){let r=0,n=e.length,a=0;for(;r<n;)e[a=Math.floor((r+n)/2)]<t?r=a+1:n=a;return n}(r,t[e+l]):function(e,t){let r=0,n=e.length,a=0;for(;r<n;)e[a=Math.floor((r+n)/2)]<=t?r=a+1:n=a;return n}(r,t[e+l])}return i}(r.data.get(a.dataId).values,r.data.get(s.dataId).values,a.shape[0],a.shape[1],s.shape[1],i);return r.makeTensorInfo(s.shape,"int32",o)}},mQ={kernelName:x.Select,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{condition:n,t:a,e:s}=t;h7([n,a,s],"select");let i=n.shape.length,o=r.data.get(n.dataId).values,l=r.data.get(a.dataId).values,u=r.data.get(s.dataId).values,h=(0,py.upcastType)(a.dtype,s.dtype),d=nR.util.makeZerosTypedArray(nR.util.sizeFromShape(a.shape),h),p=0,c=0===i||i>1||1===a.shape.length?1:nR.util.sizeFromShape(a.shape.slice(1));for(let e=0;e<o.length;e++)for(let t=0;t<c;t++)1===o[e]?d[p++]=l[e]:d[p++]=u[e];return r.makeTensorInfo(a.shape,h,d)}};var tN=tN;let m0=tN.SELU_SCALEALPHA,m1=tN.SELU_SCALE,m2=dN(x.Selu,e=>e>=0?m1*e:m0*(Math.exp(e)-1)),m3={kernelName:x.Selu,backendName:"cpu",kernelFunc:m2},m4=dN(x.Sign,e=>e<0?-1:+(e>0)),m5={kernelName:x.Sign,backendName:"cpu",kernelFunc:m4},m6=dN(x.Sin,e=>Math.sin(e)),m8={kernelName:x.Sin,backendName:"cpu",kernelFunc:m6},m7=dN(x.Sinh,e=>Math.sinh(e)),m9={kernelName:x.Sinh,backendName:"cpu",kernelFunc:m7},ge=Math.log(11920928955078125e-23)+2,gt=dN(x.Softplus,e=>{let t=Math.exp(e);return e<ge?t:e>-ge?e:Math.log(1+t)}),gr={kernelName:x.Softplus,backendName:"cpu",kernelFunc:gt};var tN=tN;let gn={kernelName:x.SpaceToBatchND,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{blockShape:s,paddings:i}=n;h7([a],"spaceToBatchND");let o=nR.util.sizeFromShape(s),l=[[0,0]];l.push(...i);for(let e=1+s.length;e<a.shape.length;++e)l.push([0,0]);let u=mD.kernelFunc({inputs:{x:a},backend:r,attrs:{paddings:l,constantValue:0}}),h=tN.getReshaped(u.shape,s,o,!1),d=tN.getPermuted(h.length,s.length,!1),p=tN.getReshapedPermuted(u.shape,s,o,!1),c=cb({inputs:{x:u},backend:r,attrs:{shape:h}}),f=pv({inputs:{x:c},backend:r,attrs:{perm:d}}),m=cb({inputs:{x:f},backend:r,attrs:{shape:p}});return r.disposeIntermediateTensorInfo(u),r.disposeIntermediateTensorInfo(c),r.disposeIntermediateTensorInfo(f),m}},ga={kernelName:x.SparseFillEmptyRows,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{indices:n,values:a,denseShape:s,defaultValue:i}=t;if(1!==s.shape.length)throw Error(`Dense shape must be a vector, saw:
        ${s.shape}`);if(2!==n.shape.length)throw Error(`Indices must be a matrix, saw:
        ${n.shape}`);if(1!==a.shape.length)throw Error(`Values must be a vector, saw:
        ${a.shape}`);if(0!==i.shape.length)throw Error(`Default value must be a scalar, saw:
        ${i.shape}`);let o=r.data.get(n.dataId).values,l=r.data.get(a.dataId).values,u=r.data.get(s.dataId).values,h=r.data.get(i.dataId).values[0],[d,p,c,f,m]=pV(o,n.shape,n.dtype,l,a.dtype,u,h);return[r.makeTensorInfo(p,n.dtype,d),r.makeTensorInfo([p[0]],a.dtype,c),r.makeTensorInfo([f.length],"bool",new Uint8Array(f.map(e=>Number(e)))),r.makeTensorInfo([m.length],n.dtype,new Int32Array(m))]}},gs={kernelName:x.SparseReshape,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{inputIndices:n,inputShape:a,newShape:s}=t;if(2!==n.shape.length)throw Error(`Input indices should be a matrix but received shape
        ${n.shape}`);if(1!==a.shape.length)throw Error(`Input shape should be a vector but received shape
        ${a.shape}`);if(1!==s.shape.length)throw Error(`Target shape should be a vector but received shape ${s.shape}`);let i=Array.from(r.data.get(a.dataId).values),o=r.data.get(n.dataId).values,l=Array.from(r.data.get(s.dataId).values),[u,h,d]=pH(o,n.shape,n.dtype,i,l);return[r.makeTensorInfo(h,n.dtype,u),r.makeTensorInfo([d.length],s.dtype,new Int32Array(d))]}},gi={kernelName:x.SparseSegmentMean,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{data:n,indices:a,segmentIds:s}=t;if(n.shape.length<1)throw Error("Data should be at least 1 dimensional but received scalar");if(1!==a.shape.length)throw Error(`Indices should be a vector but received shape
          ${a.shape}`);if(1!==s.shape.length)throw Error(`Segment ids should be a vector but received shape
          ${s.shape}`);if(a.shape[0]!==s.shape[0])throw Error("segmentIds and indices should have same size.");let i=r.data.get(n.dataId).values,o=r.data.get(a.dataId).values,l=r.data.get(s.dataId).values,[u,h]=pq(i,n.shape,n.dtype,o,l,!0);return r.makeTensorInfo(h,n.dtype,u)}},go={kernelName:x.SparseSegmentSum,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{data:n,indices:a,segmentIds:s}=t;if(n.shape.length<1)throw Error("Data should be at least 1 dimensional but received scalar");if(1!==a.shape.length)throw Error(`Indices should be a vector but received shape
         ${a.shape}`);if(1!==s.shape.length)throw Error(`Segment ids should be a vector but received shape
         ${s.shape}`);if(a.shape[0]!==s.shape[0])throw Error("segmentIds and indices should have same size.");let i=r.data.get(n.dataId).values,o=r.data.get(a.dataId).values,l=r.data.get(s.dataId).values,[u,h]=pq(i,n.shape,n.dtype,o,l);return r.makeTensorInfo(h,n.dtype,u)}};var tN=tN;let gl={kernelName:x.SparseToDense,backendName:"cpu",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{sparseIndices:s,sparseValues:i,defaultValue:o}=r,{outputShape:l}=a,{sliceRank:u,numUpdates:h,sliceSize:d,strides:p,outputSize:c}=tN.calculateShapes(i,s,l),f=n.bufferSync(s);switch(i.dtype){case"bool":t=pz(f,n.bufferSync(i),l,c,d,h,u,p,!!n.data.get(o.dataId).values[0],!1);break;case"float32":case"int32":t=pz(f,n.bufferSync(i),l,c,d,h,u,p,n.data.get(o.dataId).values[0],!1);break;case"string":t=pz(f,n.bufferSync(i),l,c,d,h,u,p,nR.util.decodeString(n.data.get(o.dataId).values[0]),!1);break;default:throw Error(`Unsupported type ${i.dtype}`)}return n.makeTensorInfo(l,t.dtype,t.values)}};var tN=tN;let gu={kernelName:x.SplitV,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{numOrSizeSplits:s,axis:i}=n,o=nR.util.parseAxisParam(i,a.shape)[0],l=tN.prepareSplitSize(a,s,o),u=Array(a.shape.length).fill(0),h=a.shape.slice();return l.map(e=>{let t=[...h];t[o]=e;let n=pG({inputs:{x:a},backend:r,attrs:{begin:u,size:t}});return u[o]+=e,n})}},gh={kernelName:x.Square,backendName:"cpu",kernelFunc:({inputs:e,backend:t})=>{let{x:r}=e;h7(r,"square");let n=t.data.get(r.dataId).values,a=new Float32Array(n.length);for(let e=0;e<n.length;++e){let t=n[e];a[e]=t*t}return{dataId:t.write(a,r.shape,r.dtype),shape:r.shape,dtype:r.dtype}}},gd=dN(x.Step,(e,t)=>isNaN(e)?NaN:e>0?1:t.alpha),gp={kernelName:x.Step,backendName:"cpu",kernelFunc:gd};var pB=tJ;let gc={kernelName:x.StridedSlice,backendName:"cpu",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{begin:i,end:o,strides:l,beginMask:u,endMask:h,ellipsisMask:d,newAxisMask:p,shrinkAxisMask:c}=a;h7(s,"stridedSlice");let{finalShapeSparse:f,finalShape:m,isIdentity:g,sliceDim0:x,isSimpleSlice:y,begin:b,end:v,strides:w}=pB.sliceInfo(s.shape,i,o,l,u,h,d,p,c);if(g)t=cb({inputs:{x:s},backend:n,attrs:{shape:m}});else if(x||y){nR.util.assert(s.shape.length>=1,()=>`Input must have rank at least 1, got: ${s.shape.length}`);let e=pB.computeOutShape(b,v,w),r=pG({inputs:{x:s},backend:n,attrs:{begin:b,size:e}});t=cb({inputs:{x:r},backend:n,attrs:{shape:m}}),n.disposeIntermediateTensorInfo(r)}else{let e=p2(f,n.bufferSync(s),w,b);t=n.makeTensorInfo(m,e.dtype,e.values)}return t}},gf={kernelName:x.StringNGrams,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{separator:a,nGramWidths:s,leftPad:i,rightPad:o,padWidth:l,preserveShortSequences:u}=n,{data:h,dataSplits:d}=t,[p,c]=p4(r.data.get(h.dataId).values,r.data.get(d.dataId).values,a,s,i,o,l,u);return[r.makeTensorInfo([p.length],"string",p),r.makeTensorInfo(d.shape,"int32",c)]}},gm={kernelName:x.StringSplit,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{skipEmpty:a}=n,{input:s,delimiter:i}=t;if("string"!==s.dtype)throw Error("Input must be of datatype string");if(1!==s.shape.length)throw Error(`Input must be a vector, got shape: ${s.shape}`);if(0!==i.shape.length)throw Error(`Delimiter must be a scalar, got shape: ${i.shape}`);let[o,l,u]=p5(r.data.get(s.dataId).values,r.data.get(i.dataId).values[0],a),h=l.length;return[r.makeTensorInfo([h,2],"int32",o),r.makeTensorInfo([h],"string",l),r.makeTensorInfo([2],"int32",new Int32Array(u))]}},gg={kernelName:x.StringToHashBucketFast,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{numBuckets:a}=n,{input:s}=t;if("string"!==s.dtype)throw Error("Input must be of datatype string");if(a<=0)throw Error("Number of buckets must be at least 1");let i=p6(r.data.get(s.dataId).values,a);return r.makeTensorInfo(s.shape,"int32",i)}},gx=dN(x.Tan,e=>Math.tan(e)),gy={kernelName:x.Tan,backendName:"cpu",kernelFunc:gx},gb=dN(x.Tanh,e=>Math.tanh(e)),gv={kernelName:x.Tanh,backendName:"cpu",kernelFunc:gb};var tN=tN;function gw(e,t,r){switch(r){case"reflect":var n,a,s,i,o=e,l=t;let u=o;if(u<0)if(l<=1)u=0;else{let e=2*l;u<e&&(u=e*Math.trunc(-u/e)+u),u=u<-l?u+e:-u-1}else if(u>l-1)if(l<=1)u=0;else{let e=2*l;(u-=e*Math.trunc(u/e))>=l&&(u=e-u-1)}return nR.util.clamp(0,u,l-1);case"wrap":let h;return n=e,a=t,(h=n)<0?a<=1?h=0:h+=a*(Math.trunc(-h/(a-1))+1):h>a-1&&(a<=1?h=0:h-=a*Math.trunc(h/(a-1))),nR.util.clamp(0,h,a-1);case"nearest":return s=e,i=t,nR.util.clamp(0,s,i-1);default:return e}}function gC(e,t,r,n,a,s,i,o,l,u,h){return 0<=o&&o<t&&0<=l&&l<r?e[i*n+o*a+l*s+u]:h}for(let e of[cI,dr,cS,cN,dv,c$,cR,cA,cE,cF,cO,cz,cM,cW,cU,cj,cX,cK,cY,cC,cZ,cJ,cQ,dS,c0,df,dE,c2,ds,c3,c8,c9,fe,ft,fr,fn,fa,fi,fl,fu,fh,fd,fp,fc,fm,fg,fx,fy,fb,fv,fw,fC,fS,co,fT,dL,fO,dM,fz,dW,fW,fU,fV,dV,dj,fH,fq,fj,fX,dJ,d1,du,fK,c5,fZ,fQ,f1,cu,d4,d8,f2,pt,f4,f8,f9,mr,mn,ma,mi,ps,mo,ml,mu,mh,md,mp,mc,pl,mf,mx,mv,pp,pf,mC,mk,mT,px,mN,mA,mF,mD,mz,cp,pI,m_,mM,mP,mB,dd,fP,mG,cf,cg,cv,mU,mV,mH,mq,mj,mX,mY,pL,mZ,mJ,mQ,m3,pP,m5,m8,m9,pU,mb,gr,gn,ga,gs,gi,go,gl,gu,pK,gh,pJ,p1,gp,gc,gf,gm,gg,ce,fk,gy,gv,{kernelName:x.TensorScatterUpdate,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r}=e,{tensor:n,indices:a,updates:s}=t,{sliceRank:i,numUpdates:o,sliceSize:l,strides:u,outputSize:h}=tN.calculateShapes(s,a,n.shape),d=r.bufferSync(a),p=r.bufferSync(s),c=r.bufferSync(n),f=pz(d,p,n.shape,h,l,o,i,u,c,!1);return r.makeTensorInfo(n.shape,f.dtype,f.values)}},{kernelName:x.Tile,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{reps:s}=n;h7(a,"tile");let i=ct(r.bufferSync(a),s);return r.makeTensorInfo(i.shape,i.dtype,i.values)}},{kernelName:x.TopK,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{k:s,sorted:i}=n;h7(a,"topk");let[o,l]=cn(r.data.get(a.dataId).values,a.shape,a.dtype,s,i);return[r.makeTensorInfo(o.shape,o.dtype,o.values),r.makeTensorInfo(l.shape,l.dtype,l.values)]}},{kernelName:x.Transform,backendName:"cpu",kernelFunc:function(e){let{inputs:t,attrs:r,backend:n}=e,{image:a,transforms:s}=t,{interpolation:i,fillMode:o,fillValue:l,outputShape:u}=r,[h,d,p,c]=a.shape,[f,m]=null!=u?u:[d,p],g=[h,f,m,c],x=nR.util.computeStrides(a.shape),y=x[0],b=x[1],v=x[2],w=nR.util.computeStrides(g),C=w[0],I=w[1],k=w[2],S=nR.util.getTypedArrayFromDType(a.dtype,nR.util.sizeFromShape(g));S.fill(l);let T=n.data.get(a.dataId).values,N=n.data.get(s.dataId).values;for(let e=0;e<h;++e){let t=1===s.shape[0]?N:N.subarray(8*e,8*e+8);for(let r=0;r<f;++r)for(let n=0;n<m;++n)for(let a=0;a<c;++a){let s,u=t[6]*n+t[7]*r+1;if(0===u)continue;let h=(t[0]*n+t[1]*r+t[2])/u,c=(t[3]*n+t[4]*r+t[5])/u,f=gw(h,p,o),m=gw(c,d,o);switch(i){case"nearest":s=gC(T,d,p,y,b,v,e,Math.round(m),Math.round(f),a,l);break;case"bilinear":s=function(e,t,r,n,a,s,i,o,l,u,h){let d=Math.floor(o),p=Math.floor(l),c=d+1,f=p+1,m=(f-l)*gC(e,t,r,n,a,s,i,d,p,u,h)+(l-p)*gC(e,t,r,n,a,s,i,d,f,u,h),g=(f-l)*gC(e,t,r,n,a,s,i,c,p,u,h)+(l-p)*gC(e,t,r,n,a,s,i,c,f,u,h);return(c-o)*m+(o-d)*g}(T,d,p,y,b,v,e,m,f,a,l);break;default:throw Error(`Error in Transform: Expect 'nearest' or 'bilinear', but got ${i}`)}S[e*C+r*I+n*k+a]=s}return n.makeTensorInfo(g,a.dtype,S)}return{dataId:n.write(S,g,a.dtype),shape:a.shape,dtype:a.dtype}}},pw,{kernelName:x.Unique,backendName:"cpu",kernelFunc:function(e){let{inputs:t,attrs:r,backend:n}=e,{axis:a}=r,{x:s}=t;h7(s,"unique");let{outputValues:i,outputShape:o,indices:l}=ca(n.data.get(s.dataId).values,a,s.shape,s.dtype);return[n.makeTensorInfo(o,s.dtype,i),n.makeTensorInfo([l.length],"int32",l)]}},{kernelName:x.Unpack,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{value:a}=t,{axis:s}=n;s<0&&(s+=a.shape.length);let i=a.shape.length,o=a.shape[s],l=Array(i-1),u=0;for(let e=0;e<i;e++)e!==s&&(l[u++]=a.shape[e]);let h=Array(i).fill(0),d=a.shape.slice();d[s]=1;let p=Array(o);for(let e=0;e<p.length;e++){h[s]=e;let t=pG({inputs:{x:a},backend:r,attrs:{begin:h,size:d}});p[e]=cb({inputs:{x:t},backend:r,attrs:{shape:l}}),r.disposeIntermediateTensorInfo(t)}return p}},{kernelName:x.UnsortedSegmentSum,backendName:"cpu",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,segmentIds:s}=t,{numSegments:i}=n;h7(a,"unsortedSegmentSum");let o=a.shape.length,l=s.shape.length,u=[],h=[],d=o-l,p=s;for(let e=0;e<d;++e){let t=fL({inputs:{input:p},backend:r,attrs:{dim:e+1}});p=t,h.push(t)}for(let e=0;e<i;++e){let t=nR.util.createScalarValue(e,"int32"),n=r.makeTensorInfo([],"int32",t),s=dO({inputs:{a:n,b:p},backend:r}),i=dc({inputs:{x:s},backend:r,attrs:{dtype:"float32"}}),o=pd({inputs:{a:i,b:a},backend:r}),l=fI({inputs:{x:o},backend:r,attrs:{axis:0,keepDims:!1}});u.push(l),h.push(n),h.push(s),h.push(i),h.push(o),h.push(l)}let c=mE({inputs:u,backend:r,attrs:{axis:0}});return h.forEach(e=>r.disposeIntermediateTensorInfo(e)),c}},mR])(0,rc.registerKernel)(e);e.s([],18538);var gI=e.i(23221),gI=gI,gI=gI;let gk={},gS={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function gT(e,t){gk[e]=t}function gN(e,t){if(!(e in gk)||null!=t){let r=function(e,t){if(1!==e&&2!==e)throw Error("Cannot get WebGL rendering context, WebGL is disabled.");let r=null==t?function(e){if(!(0,nN.env)().getBool("IS_SAFARI")&&"u">typeof OffscreenCanvas&&2===e)return new OffscreenCanvas(300,150);if("u">typeof document)return document.createElement("canvas");throw Error("Cannot create a canvas in this context")}(e):t;return(r.addEventListener("webglcontextlost",t=>{t.preventDefault(),delete gk[e]},!1),(0,nN.env)().getBool("SOFTWARE_WEBGL_ENABLED")&&(gS.failIfMajorPerformanceCaveat=!1),1===e)?r.getContext("webgl",gS)||r.getContext("experimental-webgl",gS):r.getContext("webgl2",gS)}(e,t);if(null===r)return console.log("Could not get context for WebGL version",e),null;gk[e]=r}let r=gk[e];return null==r||r.isContextLost()?(delete gk[e],gN(e)):(r.disable(r.DEPTH_TEST),r.disable(r.STENCIL_TEST),r.disable(r.BLEND),r.disable(r.DITHER),r.disable(r.POLYGON_OFFSET_FILL),r.disable(r.SAMPLE_COVERAGE),r.enable(r.SCISSOR_TEST),r.enable(r.CULL_FACE),r.cullFace(r.BACK),gk[e])}function g$(e){let t=Math.ceil(nR.util.sizeFromShape(e)/4);return nR.util.sizeToSquarishShape(t)}function gR(e,t){return[Math.max(1,Math.ceil(t/2)),Math.max(1,Math.ceil(e/2))]}function gA(e,t){let r,n,a,s,i,o,l,u,h,d;return 2===(0,nN.env)().getNumber("WEBGL_VERSION")?(r=e.R32F,n=e.R16F,a=e.RGBA16F,s=e.RGBA32F,i=e.RED,l=4,u=1,h=e.HALF_FLOAT,d=e.FLOAT,o=e.RGBA8):(r=e.RGBA,n=e.RGBA,a=e.RGBA,s=e.RGBA,i=e.RGBA,l=4,u=4,h=null!=t?t.HALF_FLOAT_OES:null,d=e.FLOAT,o=e.RGBA),{internalFormatFloat:r,internalFormatHalfFloat:n,internalFormatPackedHalfFloat:a,internalFormatPackedFloat:s,textureFormatFloat:i,downloadTextureFormat:o,downloadUnpackNumChannels:l,defaultNumChannels:u,textureTypeHalfFloat:h,textureTypeFloat:d}}function gE(e,t){let r=t();return(0,nN.env)().getBool("DEBUG")&&function(e){let t=e.getError();if(t!==e.NO_ERROR)throw Error("WebGL Error: "+gD(e,t))}(e),r}function gF(e){return!!((0,nN.env)().getBool("WEBGL_RENDER_FLOAT32_ENABLED")||0===e||596e-10<Math.abs(e)&&65504>Math.abs(e))}function gD(e,t){switch(t){case e.NO_ERROR:return"NO_ERROR";case e.INVALID_ENUM:return"INVALID_ENUM";case e.INVALID_VALUE:return"INVALID_VALUE";case e.INVALID_OPERATION:return"INVALID_OPERATION";case e.INVALID_FRAMEBUFFER_OPERATION:return"INVALID_FRAMEBUFFER_OPERATION";case e.OUT_OF_MEMORY:return"OUT_OF_MEMORY";case e.CONTEXT_LOST_WEBGL:return"CONTEXT_LOST_WEBGL";default:return`Unknown error code ${t}`}}function gO(e,t){return g2(e,()=>e.getExtension(t),'Extension "'+t+'" not supported on this browser.')}function gL(e,t){let r=g2(e,()=>e.createShader(e.VERTEX_SHADER),"Unable to create vertex WebGLShader.");if(gE(e,()=>e.shaderSource(r,t)),gE(e,()=>e.compileShader(r)),!1===e.getShaderParameter(r,e.COMPILE_STATUS))throw console.log(e.getShaderInfoLog(r)),Error("Failed to compile vertex shader.");return r}function gz(e,t){let r=g2(e,()=>e.createShader(e.FRAGMENT_SHADER),"Unable to create fragment WebGLShader.");if(gE(e,()=>e.shaderSource(r,t)),gE(e,()=>e.compileShader(r)),(0,nN.env)().get("ENGINE_COMPILE_ONLY"))return r;if(!1===e.getShaderParameter(r,e.COMPILE_STATUS))throw gM(t,e.getShaderInfoLog(r)),Error("Failed to compile fragment shader.");return r}(o=c||(c={}))[o.DENSE=0]="DENSE",o[o.SHARED_BATCH=1]="SHARED_BATCH",(l=f||(f={}))[l.RENDER=0]="RENDER",l[l.UPLOAD=1]="UPLOAD",l[l.PIXELS=2]="PIXELS",l[l.DOWNLOAD=3]="DOWNLOAD",(u=m||(m={}))[u.UNPACKED_FLOAT16=0]="UNPACKED_FLOAT16",u[u.UNPACKED_FLOAT32=1]="UNPACKED_FLOAT32",u[u.PACKED_4X1_UNSIGNED_BYTE=2]="PACKED_4X1_UNSIGNED_BYTE",u[u.PACKED_2X2_FLOAT32=3]="PACKED_2X2_FLOAT32",u[u.PACKED_2X2_FLOAT16=4]="PACKED_2X2_FLOAT16";let g_=/ERROR: [0-9]+:([0-9]+):/g;function gM(e,t){let r=g_.exec(t);if(null==r){console.log(`Couldn't parse line number in error: ${t}`),console.log(e);return}let n=+r[1],a=e.split("\n"),s=a.length.toString().length+2,i=a.map((e,t)=>nR.util.rightPad((t+1).toString(),s)+e),o=0;for(let e=0;e<i.length;e++)o=Math.max(i[e].length,o);let l=i.slice(0,n-1),u=i.slice(n-1,n),h=i.slice(n);console.log(l.join("\n")),console.log(t.split("\n")[0]),console.log(`%c ${nR.util.rightPad(u[0],o)}`,"border:1px solid red; background-color:#e3d2d2; color:#a61717"),console.log(h.join("\n"))}function gP(e){return g2(e,()=>e.createProgram(),"Unable to create WebGLProgram.")}function gB(e,t){if(gE(e,()=>e.linkProgram(t)),!(0,nN.env)().get("ENGINE_COMPILE_ONLY")&&!1===e.getProgramParameter(t,e.LINK_STATUS))throw console.log(e.getProgramInfoLog(t)),Error("Failed to link vertex and fragment shaders.")}function gW(e,t){if(gE(e,()=>e.validateProgram(t)),!1===e.getProgramParameter(t,e.VALIDATE_STATUS))throw console.log(e.getProgramInfoLog(t)),Error("Shader program validation failed.")}function gG(e,t){let r=g2(e,()=>e.createBuffer(),"Unable to create WebGLBuffer");return gE(e,()=>e.bindBuffer(e.ARRAY_BUFFER,r)),gE(e,()=>e.bufferData(e.ARRAY_BUFFER,t,e.STATIC_DRAW)),r}function gU(e,t){let r=g2(e,()=>e.createBuffer(),"Unable to create WebGLBuffer");return gE(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,r)),gE(e,()=>e.bufferData(e.ELEMENT_ARRAY_BUFFER,t,e.STATIC_DRAW)),r}function gV(e){return g2(e,()=>e.createTexture(),"Unable to create WebGLTexture.")}function gH(e,t){let r=(0,nN.env)().getNumber("WEBGL_MAX_TEXTURE_SIZE");if(e<=0||t<=0)throw Error(`Requested texture size [${e}x${t}] is invalid.`);if(e>r||t>r)throw Error(`Requested texture size [${e}x${t}] greater than WebGL maximum on this browser / GPU [${r}x${r}].`)}function gq(e){return g2(e,()=>e.createFramebuffer(),"Unable to create WebGLFramebuffer.")}function gj(e,t,r,n,a,s,i){let o=e.getAttribLocation(t,r);return -1!==o&&(gE(e,()=>e.bindBuffer(e.ARRAY_BUFFER,n)),gE(e,()=>e.vertexAttribPointer(o,a,e.FLOAT,!1,s,i)),gE(e,()=>e.enableVertexAttribArray(o)),!0)}function gX(e,t,r){g3(e,r),gE(e,()=>e.activeTexture(e.TEXTURE0+r)),gE(e,()=>e.bindTexture(e.TEXTURE_2D,t))}function gK(e,t,r){return g2(e,()=>e.getUniformLocation(t,r),'uniform "'+r+'" not present in program.')}function gY(e,t,r){return e.getUniformLocation(t,r)}function gZ(e,t,r,n){gE(e,()=>gX(e,t,n)),gE(e,()=>e.uniform1i(r,n))}function gJ(e,t,r){gE(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,r)),gE(e,()=>e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,t,0))}function gQ(e,t){gE(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,t)),gE(e,()=>e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,null,0))}function g0(e){let t=e.checkFramebufferStatus(e.FRAMEBUFFER);if(t!==e.FRAMEBUFFER_COMPLETE)throw Error("Error binding framebuffer: "+g1(e,t))}function g1(e,t){switch(t){case e.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_ATTACHMENT";case e.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";case e.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return"FRAMEBUFFER_INCOMPLETE_DIMENSIONS";case e.FRAMEBUFFER_UNSUPPORTED:return"FRAMEBUFFER_UNSUPPORTED";default:return`unknown error ${t}`}}function g2(e,t,r){let n=gE(e,()=>t());if(null==n)throw Error(r);return n}function g3(e,t){let r=e.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,n=t+e.TEXTURE0;if(n<e.TEXTURE0||n>r){let e=`[gl.TEXTURE0, gl.TEXTURE${r}]`;throw Error(`textureUnit must be in ${e}.`)}}function g4(e,t=2){return nR.util.sizeFromShape(e.slice(0,e.length-t))}function g5(e){if(0===e.length)throw Error("Cannot get rows and columns of an empty shape array.");return[e.length>1?e[e.length-2]:1,e[e.length-1]]}function g6(e){let t=[1,1,1];return 0!==e.length&&(1!==e.length||1!==e[0])&&(t=[g4(e),...g5(e)]),t}function g8(e,t=!1){let r=(0,nN.env)().getNumber("WEBGL_MAX_TEXTURE_SIZE"),n=(0,nN.env)().getNumber("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE");n===1/0&&(0,nN.env)().getBool("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE")&&(n=r/2),t&&(r*=2,n*=2,1===(e=e.map((t,r)=>r>=e.length-2?nR.util.nearestLargerEven(e[r]):e[r])).length&&(e=[2,e[0]])),2!==e.length&&(e=nR.util.squeezeShape(e).newShape);let a=nR.util.sizeFromShape(e),s=null;e.length<=1&&a<=r?s=[1,a]:2===e.length&&e[0]<=r&&e[1]<=r?s=e:3===e.length&&e[0]*e[1]<=r&&e[2]<=r?s=[e[0]*e[1],e[2]]:3===e.length&&e[0]<=r&&e[1]*e[2]<=r?s=[e[0],e[1]*e[2]]:4===e.length&&e[0]*e[1]*e[2]<=r&&e[3]<=r?s=[e[0]*e[1]*e[2],e[3]]:4===e.length&&e[0]<=r&&e[1]*e[2]*e[3]<=r&&(s=[e[0],e[1]*e[2]*e[3]]);let i=null!=s&&Math.max(...s)>n&&Math.min(...s)<=(t?2:1)&&Math.min(...s)>0;if(null==s||i)if(t){let t=g4(e),r=2,n=2;e.length&&([r,n]=g5(e)),a=r/2*t*(n/2),s=nR.util.sizeToSquarishShape(a).map(e=>2*e)}else s=nR.util.sizeToSquarishShape(a);return s}function g7(e,t){if(e=e.slice(-2),t=t.slice(-2),nR.util.arraysEqual(e,t)||!e.length||!t.length||0===e[0]||0===e[1]||0===t[0]||0===t[1])return!0;if(e.length!==t.length){let r=e[e.length-1],n=t[t.length-1];if(r===n||r%2==0&&n%2==0&&(1===e[0]||1===t[0]))return!0}return e[1]===t[1]&&e[0]%2==0&&t[0]%2==0}function g9(e){if(null==r){let t=gN(e);r=t.getParameter(t.MAX_TEXTURE_SIZE)}return r}function xe(e){if(null==n){let t=gN(e);n=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,n)}function xt(e){if(0===e)return 0;let t=gN(e);return xr(t,"EXT_disjoint_timer_query_webgl2")&&2===e?2:+!!xr(t,"EXT_disjoint_timer_query")}function xr(e,t){return null!=e.getExtension(t)}function xn(e){try{let t=gN(e);if(null!=t)return!0}catch(e){console.log("Error when getting WebGL context: ",e)}return!1}function xa(e){if(0===e)return!1;let t=gN(e);if(1===e){if(!xr(t,"OES_texture_float"))return!1}else if(!xr(t,"EXT_color_buffer_float"))return!1;return xi(t)}function xs(e){if(0===e)return!1;let t=gN(e);if(1===e){if(!xr(t,"OES_texture_float")||!xr(t,"WEBGL_color_buffer_float"))return!1}else{if(xr(t,"EXT_color_buffer_float"))return xi(t);let e="EXT_color_buffer_half_float";if(xr(t,e)){var r;let n,a,s,i,o=t.getExtension(e);return n=gA(r=t,o),a=r.createTexture(),r.bindTexture(r.TEXTURE_2D,a),r.texImage2D(r.TEXTURE_2D,0,n.internalFormatHalfFloat,1,1,0,n.textureFormatFloat,n.textureTypeHalfFloat,null),s=r.createFramebuffer(),r.bindFramebuffer(r.FRAMEBUFFER,s),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,a,0),i=r.checkFramebufferStatus(r.FRAMEBUFFER)===r.FRAMEBUFFER_COMPLETE,r.bindTexture(r.TEXTURE_2D,null),r.bindFramebuffer(r.FRAMEBUFFER,null),r.deleteTexture(a),r.deleteFramebuffer(s),i}return!1}return xi(t)}function xi(e){let t=gA(e),r=e.createTexture();e.bindTexture(e.TEXTURE_2D,r),e.texImage2D(e.TEXTURE_2D,0,t.internalFormatFloat,1,1,0,t.textureFormatFloat,t.textureTypeFloat,null);let n=e.createFramebuffer();e.bindFramebuffer(e.FRAMEBUFFER,n),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,r,0);let a=e.checkFramebufferStatus(e.FRAMEBUFFER)===e.FRAMEBUFFER_COMPLETE;return e.bindTexture(e.TEXTURE_2D,null),e.bindFramebuffer(e.FRAMEBUFFER,null),e.deleteTexture(r),e.deleteFramebuffer(n),a}function xo(e){return 2===e&&null!=gN(e).fenceSync}function xl(e,t){Array.isArray(e)||(e=[e]),e.forEach(e=>{null!=e&&nR.util.assert("complex64"!==e.dtype,()=>`${t} does not support complex64 tensors in the WebGL backend.`)})}e.s(["assertNotComplex",0,xl,"bindCanvasToFramebuffer",0,function(e){gE(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,null)),gE(e,()=>e.viewport(0,0,e.canvas.width,e.canvas.height)),gE(e,()=>e.scissor(0,0,e.canvas.width,e.canvas.height))},"bindColorTextureToFramebuffer",0,gJ,"bindTextureToProgramUniformSampler",0,gZ,"bindTextureUnit",0,gX,"bindVertexBufferToProgramAttribute",0,gj,"callAndCheck",0,gE,"canBeRepresented",0,gF,"createFragmentShader",0,gz,"createFramebuffer",0,gq,"createProgram",0,gP,"createStaticIndexBuffer",0,gU,"createStaticVertexBuffer",0,gG,"createTexture",0,gV,"createVertexShader",0,gL,"getBatchDim",0,g4,"getExtensionOrThrow",0,gO,"getFramebufferErrorMessage",0,g1,"getMaxTexturesInShader",0,xe,"getNumChannels",0,function(){return 2===(0,nN.env)().getNumber("WEBGL_VERSION")?1:4},"getProgramUniformLocation",0,gY,"getProgramUniformLocationOrThrow",0,gK,"getRowsCols",0,g5,"getShapeAs3D",0,g6,"getTextureShapeFromLogicalShape",0,g8,"getWebGLDisjointQueryTimerVersion",0,xt,"getWebGLErrorMessage",0,gD,"getWebGLMaxTextureSize",0,g9,"hasExtension",0,xr,"isCapableOfRenderingToFloatTexture",0,xa,"isDownloadFloatTextureEnabled",0,xs,"isReshapeFree",0,g7,"isWebGLFenceEnabled",0,xo,"isWebGLVersionEnabled",0,xn,"linkProgram",0,gB,"logShaderSourceAndInfoLog",0,gM,"resetMaxTextureSize",0,function(){r=null},"resetMaxTexturesInShader",0,function(){n=null},"unbindColorTextureFromFramebuffer",0,gQ,"unbindTextureUnit",0,function(e,t){g3(e,t),gE(e,()=>e.activeTexture(e.TEXTURE0+t)),gE(e,()=>e.bindTexture(e.TEXTURE_2D,null))},"validateFramebuffer",0,g0,"validateProgram",0,gW,"validateTextureSize",0,gH],71065);let xu=(0,nN.env)();xu.registerFlag("HAS_WEBGL",()=>xu.getNumber("WEBGL_VERSION")>0),xu.registerFlag("WEBGL_VERSION",()=>xn(2)?2:+!!xn(1)),xu.registerFlag("WEBGL_CHECK_NUMERICAL_PROBLEMS",()=>!1),xu.registerFlag("WEBGL_BUFFER_SUPPORTED",()=>2===xu.get("WEBGL_VERSION")),xu.registerFlag("WEBGL_CPU_FORWARD",()=>!0),xu.registerFlag("WEBGL_FORCE_F16_TEXTURES",()=>!1),xu.registerFlag("WEBGL_PACK",()=>xu.getBool("HAS_WEBGL")),xu.registerFlag("WEBGL_PACK_NORMALIZATION",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_PACK_CLIP",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_PACK_DEPTHWISECONV",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_PACK_BINARY_OPERATIONS",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_PACK_UNARY_OPERATIONS",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_PACK_ARRAY_OPERATIONS",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_PACK_IMAGE_OPERATIONS",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_PACK_REDUCE",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_LAZILY_UNPACK",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_CONV_IM2COL",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_PACK_CONV2DTRANSPOSE",()=>xu.getBool("WEBGL_PACK")),xu.registerFlag("WEBGL_MAX_TEXTURE_SIZE",()=>g9(xu.getNumber("WEBGL_VERSION"))),xu.registerFlag("WEBGL_MAX_TEXTURES_IN_SHADER",()=>xe(xu.getNumber("WEBGL_VERSION"))),xu.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION",()=>{let e=xu.getNumber("WEBGL_VERSION");return 0===e?0:xt(e)}),xu.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE",()=>xu.getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0&&!gI.isMobile()),xu.registerFlag("WEBGL_RENDER_FLOAT32_CAPABLE",()=>xa(xu.getNumber("WEBGL_VERSION"))),xu.registerFlag("WEBGL_RENDER_FLOAT32_ENABLED",()=>!xu.getBool("WEBGL_FORCE_F16_TEXTURES")&&xu.getBool("WEBGL_RENDER_FLOAT32_CAPABLE")),xu.registerFlag("WEBGL_DOWNLOAD_FLOAT_ENABLED",()=>xs(xu.getNumber("WEBGL_VERSION"))),xu.registerFlag("WEBGL_FENCE_API_ENABLED",()=>xo(xu.getNumber("WEBGL_VERSION"))),xu.registerFlag("WEBGL_SIZE_UPLOAD_UNIFORM",()=>4*!!xu.getBool("WEBGL_RENDER_FLOAT32_ENABLED")),xu.registerFlag("WEBGL_DELETE_TEXTURE_THRESHOLD",()=>-1,e=>{if("number"!=typeof e)throw Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be a number but got ${e}.`);if(e<0&&-1!==e)throw Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${e}.`)}),xu.registerFlag("WEBGL_FLUSH_THRESHOLD",()=>gI.isMobile()?1:-1,e=>{if("number"!=typeof e)throw Error(`WEBGL_FLUSH_THRESHOLD must be a number but got ${e}.`);if(e<0&&-1!==e)throw Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${e}.`)}),xu.registerFlag("CPU_HANDOFF_SIZE_THRESHOLD",()=>128),xu.registerFlag("WEBGL_USE_SHAPES_UNIFORMS",()=>!1),xu.registerFlag("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD",()=>1e5),xu.registerFlag("TOPK_K_CPU_HANDOFF_THRESHOLD",()=>128),xu.registerFlag("WEBGL_EXP_CONV",()=>!1),xu.registerFlag("SOFTWARE_WEBGL_ENABLED",()=>xu.getBool("IS_TEST")),xu.registerFlag("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE",()=>1/0),xu.registerFlag("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE",()=>!1),xu.registerFlag("WEBGL2_ISNAN_CUSTOM",()=>!1),xu.registerFlag("ENGINE_COMPILE_ONLY",()=>!1);var tN=tN,xh=h6,h8=h8;function xd(){let e,t,r,n,a,s,i,o,l,u;return 2===(0,nN.env)().getNumber("WEBGL_VERSION")?(e="#version 300 es",t="in",r="out",n="in",a="texture",s="outputColor",i="out vec4 outputColor;",o=(0,nN.env)().getBool("WEBGL2_ISNAN_CUSTOM")?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:"",l="",u=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(e="",t="attribute",r="varying",n="varying",a="texture2D",s="gl_FragColor",i="",o=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,l=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,u=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:e,attribute:t,varyingVs:r,varyingFs:n,texture2D:a,output:s,defineOutput:i,defineSpecialNaN:o,defineSpecialInf:l,defineRound:u}}var tN=tN,tN=tN;function xp(e,t,r="index"){let n=nR.util.computeStrides(t);return n.map((t,a)=>{let s=`int ${e[a]} = ${r} / ${t}`,i=a===n.length-1?`int ${e[a+1]} = ${r} - ${e[a]} * ${t}`:`index -= ${e[a]} * ${t}`;return`${s}; ${i};`}).join("")}function xc(e,t,r="index"){let n=nR.util.computeStrides(t);return n.map((t,a)=>{let s=`int ${e[a]} = ${r} / outShapeStrides[${a}]`,i=a===n.length-1?`int ${e[a+1]} = ${r} - ${e[a]} * outShapeStrides[${a}]`:`index -= ${e[a]} * outShapeStrides[${a}]`;return`${s}; ${i};`}).join("")}function xf(e){let t=nR.util.computeStrides(e).map(e=>e.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${t[0]} + coords.y * ${t[1]} + coords.z;
  }
`}function xm(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}let xg=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`,{getBroadcastDims:xx}=tN,xy=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,xb=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,xv=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,xw=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function xC(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function xI(e){return`offset${e}`}function xk(e){let t=e.name,r=nR.util.sizeFromShape(e.shapeInfo.logicalShape);return r<2?`return ${t};`:`
    for (int i = 0; i < ${r}; i++) {
      if (i == index) {
        return ${t}[i];
      }
    }
  `}function xS(e){if(e<=1)return"int";if(2===e)return"ivec2";if(3===e)return"ivec3";if(4===e)return"ivec4";if(5===e)return"ivec5";else if(6===e)return"ivec6";else throw Error(`GPU for rank ${e} is not yet supported`)}function xT(e,t,r){let{newShape:n,keptDims:a}=nR.util.squeezeShape(t),s=t.length,i=e&&3===s&&1===t[0],o=i?t.slice(1):n,l=!e&&s>1&&!nR.util.arraysEqual(t,r)&&n.length<s||i,u=l?o:t;return{useSqueezeShape:l,uniformShape:u,keptDims:a}}function xN(e,t){let r=JSON.parse(JSON.stringify(e));return r.shapeInfo.logicalShape=t,r}function x$(e,t){return t.map(t=>e[t]).join(", ")}function xR(e,t,r){let n,a,s,i=[],o=[],l=null,u=null;for(let n of(u=e.getUniformLocation(r,"NAN",!1),1===(0,nN.env)().getNumber("WEBGL_VERSION")&&(l=e.getUniformLocation(r,"INFINITY",!1)),t.variableNames)){let a={name:n,uniform:e.getUniformLocation(r,n,!1),offset:e.getUniformLocation(r,`offset${n}`,!1)};t.enableShapeUniforms&&(a.shape=e.getUniformLocation(r,`${n}Shape`,!1),a.texShape=e.getUniformLocation(r,`${n}TexShape`,!1)),i.push(a)}if(t.enableShapeUniforms&&(n=e.getUniformLocation(r,"outShape",!1),s=e.getUniformLocation(r,"outShapeStrides",!1),a=e.getUniformLocation(r,"outTexShape",!1)),t.customUniforms)for(let n of t.customUniforms)o.push(e.getUniformLocation(r,n.name,!1));return{variablesLocations:i,customUniformLocations:o,infLoc:l,nanLoc:u,outShapeLocation:n,outShapeStridesLocation:s,outTexShapeLocation:a}}function xA(e,t){if(e.length!==t.length)throw Error(`Binary was compiled with ${e.length} inputs, but was executed with ${t.length} inputs`);e.forEach((e,r)=>{let n=e.logicalShape,a=t[r],s=a.shape;if(!nR.util.arraysEqual(n,s))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${n} and ${s} must match`);if(e.isUniform&&a.isUniform)return;let i=e.texShape,o=a.isUniform?null:a.texData.texShape;if(!nR.util.arraysEqual(i,o))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${i} and ${o} must match`)})}function xE(e){return(0,nN.env)().getBool("WEBGL_USE_SHAPES_UNIFORMS")&&e<=4}class xF{constructor(e){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=c.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const t=xd();this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?xc(["r","c","d"],e):xp(["r","c","d"],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${t.output} = result;
      }
    `}}class xD{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=c.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const t=xd();this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?xc(["r","c","d"],e):xp(["r","c","d"],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${t.output} = result;
      }
    `}}class xO{constructor(e){this.variableNames=["A"],this.outTexUsage=f.DOWNLOAD;const t=xd();this.outputShape=e,this.userCode=`
      ${xg}

      void main() {
        float x = getAAtOutCoords();
        ${t.output} = encode_float(x);
      }
    `}}class xL{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=f.DOWNLOAD;const t=xd();this.outputShape=e,this.userCode=`
      ${xg}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${t.output} = encode_float(x);
      }
    `}}let xz={R:0,G:1,B:2,A:3};class x_{constructor(e,t=!1,r="RGBA"){this.variableNames=["A"],this.customUniforms=[{name:"texShape",type:"ivec2"}];const n=xd();this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length);let a="result";t&&(a="floor(result * 255. + 0.5)");let s="";for(let e=0;e<r.length;e++){const t=r[e];s+=`
          if(offset == ${e}) {
            result = values[${xz[t]}];
          }`}this.userCode=`
      ${this.enableShapeUniforms?xm():xf(e)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${r.length});

        flatIndex = idiv(flatIndex, ${r.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${n.texture2D}(A, uv);
          ${s}
        }
        ${n.output} = vec4(${a}, 0., 0., 0.);
      }
    `}}class xM{constructor(e,t=!1){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:"texShape",type:"ivec2"}];const r=xd();this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length);let n="",a="result";t&&(a="floor(result * 255. + 0.5)");for(let t=0;t<=1;t++)for(let a=0;a<=1;a++){const s=2*t+a;n+=`
          localCoords = coords;
          if(localCoords[2] + ${a} < ${this.enableShapeUniforms?"outShape[2]":`${e[2]}`}) {
          localCoords[2] += ${a};
          if (localCoords[1] + ${t} < ${this.enableShapeUniforms?"outShape[1]":`${e[1]}`}) {
            localCoords[1] += ${t};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${r.texture2D}(A, uv);

            if (offset == 0) {
              result[${s}] = values[0];
            } else if (offset == 1) {
              result[${s}] = values[1];
            } else if (offset == 2) {
              result[${s}] = values[2];
            } else {
              result[${s}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?xm():xf(e)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${n}

          ${r.output} = ${a};
        }
    `}}function xP(e){let t=xd();return gL(e,`${t.version}
    precision highp float;
    ${t.attribute} vec3 clipSpacePos;
    ${t.attribute} vec2 uv;
    ${t.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`)}function xB(e){return gG(e,new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]))}function xW(e){return gU(e,new Uint16Array([0,1,2,2,1,3]))}function xG(e,t,r,n,a,s){gH(t,r);let i=gV(e),o=e.TEXTURE_2D;return gE(e,()=>e.bindTexture(o,i)),gE(e,()=>e.texParameteri(o,e.TEXTURE_WRAP_S,e.CLAMP_TO_EDGE)),gE(e,()=>e.texParameteri(o,e.TEXTURE_WRAP_T,e.CLAMP_TO_EDGE)),gE(e,()=>e.texParameteri(o,e.TEXTURE_MIN_FILTER,e.NEAREST)),gE(e,()=>e.texParameteri(o,e.TEXTURE_MAG_FILTER,e.NEAREST)),1===(0,nN.env)().getNumber("WEBGL_VERSION")?gE(e,()=>e.texImage2D(o,0,n,t,r,0,a,s,null)):gE(e,()=>e.texStorage2D(o,1,n,t,r)),gE(e,()=>e.bindTexture(e.TEXTURE_2D,null)),{texture:i,texShape:[r,t]}}function xU(e){return e.internalFormatFloat}function xV(e,t,r,n){let[a,s]=[r,t];return xG(e,a,s,xU(n),n.textureFormatFloat,e.FLOAT)}function xH(e){return e.internalFormatHalfFloat}function xq(e,t,r,n){let[a,s]=[r,t];return xG(e,a,s,xH(n),n.textureFormatFloat,n.textureTypeHalfFloat)}function xj(e){return e.downloadTextureFormat}function xX(e,t,r,n){let[a,s]=[r,t];return xG(e,a,s,xj(n),e.RGBA,e.UNSIGNED_BYTE)}function xK(e){return e.internalFormatPackedFloat}function xY(e,t,r,n){let[a,s]=gR(t,r);return xG(e,a,s,xK(n),e.RGBA,e.FLOAT)}function xZ(e){return e.internalFormatPackedHalfFloat}function xJ(e,t,r,n){let[a,s]=gR(t,r);return xG(e,a,s,xZ(n),e.RGBA,n.textureTypeHalfFloat)}function xQ(e,t,r){return gE(e,()=>e.bindBuffer(e.ARRAY_BUFFER,r)),gj(e,t,"clipSpacePos",r,3,20,0)&&gj(e,t,"uv",r,2,20,12)}function x0(e,t,r,n,a,s){let i,o,l;gE(e,()=>e.bindTexture(e.TEXTURE_2D,t)),a instanceof Uint8Array?(i=new Uint8Array(r*n*4),o=e.UNSIGNED_BYTE,l=e.RGBA):(i=new Float32Array(r*n*4),o=e.FLOAT,l=s.internalFormatPackedFloat),i.set(a),2===(0,nN.env)().getNumber("WEBGL_VERSION")?gE(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,r,n,e.RGBA,o,i)):gE(e,()=>e.texImage2D(e.TEXTURE_2D,0,l,r,n,0,e.RGBA,o,i)),gE(e,()=>e.bindTexture(e.TEXTURE_2D,null))}function x1(e,t,r){gE(e,()=>e.bindTexture(e.TEXTURE_2D,t)),r.data instanceof Uint8Array?2===(0,nN.env)().getNumber("WEBGL_VERSION")?gE(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,r.width,r.height,e.RGBA,e.UNSIGNED_BYTE,r.data)):gE(e,()=>e.texImage2D(e.TEXTURE_2D,0,e.RGBA,r.width,r.height,0,e.RGBA,e.UNSIGNED_BYTE,r.data)):2===(0,nN.env)().getNumber("WEBGL_VERSION")?gE(e,()=>e.texSubImage2D(e.TEXTURE_2D,0,0,0,e.RGBA,e.UNSIGNED_BYTE,r)):gE(e,()=>e.texImage2D(e.TEXTURE_2D,0,e.RGBA,e.RGBA,e.UNSIGNED_BYTE,r)),gE(e,()=>e.bindTexture(e.TEXTURE_2D,null))}function x2(e,t,r,n){let a=e.createBuffer();gE(e,()=>e.bindBuffer(e.PIXEL_PACK_BUFFER,a));let s=16*t*r;return gE(e,()=>e.bufferData(e.PIXEL_PACK_BUFFER,s,e.STREAM_READ)),gE(e,()=>e.readPixels(0,0,r,t,e.RGBA,e.FLOAT,0)),gE(e,()=>e.bindBuffer(e.PIXEL_PACK_BUFFER,null)),a}function x3(e,t,r){let n=new Float32Array(r);return e.bindBuffer(e.PIXEL_PACK_BUFFER,t),e.getBufferSubData(e.PIXEL_PACK_BUFFER,0,n),e.bindBuffer(e.PIXEL_PACK_BUFFER,null),n}function x4(e,t,r,n){let[a,s]=[r,t],i=new Uint8Array(t*r*4);return gE(e,()=>e.readPixels(0,0,a,s,n.downloadTextureFormat,e.UNSIGNED_BYTE,i)),new Float32Array(i.buffer)}function x5(e,t,r,n,a,s,i,o){let l=new Float32Array(function(e,t){let[r,n]=gR(e,t);return r*n*4}(s,i));return e.bindBuffer(e.PIXEL_PACK_BUFFER,t),e.getBufferSubData(e.PIXEL_PACK_BUFFER,0,l),e.bindBuffer(e.PIXEL_PACK_BUFFER,null),l}function x6(e,t,r){let n=new Float32Array(t*r*4);return gE(e,()=>e.readPixels(0,0,r,t,e.RGBA,e.FLOAT,n)),n}e.s(["bindVertexProgramAttributeStreams",0,xQ,"createBufferFromOutputTexture",0,x2,"createFloat16MatrixTexture",0,xq,"createFloat16PackedMatrixTexture",0,xJ,"createFloat32MatrixTexture",0,xV,"createIndexBuffer",0,xW,"createPackedMatrixTexture",0,xY,"createUnsignedBytesMatrixTexture",0,xX,"createVertexBuffer",0,xB,"createVertexShader",0,xP,"downloadByteEncodedFloatMatrixFromOutputTexture",0,x4,"downloadFloat32MatrixFromBuffer",0,x3,"downloadMatrixFromPackedOutputTexture",0,x6,"downloadPackedMatrixFromBuffer",0,x5,"getInternalFormatForFloat16MatrixTexture",0,xH,"getInternalFormatForFloat16PackedMatrixTexture",0,xZ,"getInternalFormatForFloat32MatrixTexture",0,xU,"getInternalFormatForPackedMatrixTexture",0,xK,"getInternalFormatForUnsignedBytesMatrixTexture",0,xj,"uploadDenseMatrixToTexture",0,x0,"uploadPixelDataToTexture",0,x1],86136);class x8{constructor(e){this.outputTexture=null,this.program=null,this.disposed=!1,this.itemsToPoll=[];const t=(0,nN.env)().getNumber("WEBGL_VERSION");if(null!=e?(this.gl=e,gT(t,e)):this.gl=gN(t),e=this.gl,2===(0,nN.env)().getNumber("WEBGL_VERSION")){const t=e;this.createVertexArray=()=>gE(t,()=>t.createVertexArray()),this.bindVertexArray=e=>gE(t,()=>t.bindVertexArray(e)),this.deleteVertexArray=e=>gE(t,()=>t.deleteVertexArray(e)),this.getVertexArray=()=>gE(t,()=>t.getParameter(t.VERTEX_ARRAY_BINDING))}else if(null!=e){const t=e.getExtension("OES_vertex_array_object");if(null==t)throw Error("All WebGL1 implementations are expected to offer OES_vertex_array_object.");this.createVertexArray=()=>gE(e,()=>t.createVertexArrayOES()),this.bindVertexArray=r=>gE(e,()=>t.bindVertexArrayOES(r)),this.deleteVertexArray=r=>gE(e,()=>t.deleteVertexArrayOES(r)),this.getVertexArray=()=>gE(e,()=>e.getParameter(t.VERTEX_ARRAY_BINDING_OES))}let r="WEBGL_color_buffer_float";const n="EXT_color_buffer_half_float";if(this.parallelCompilationExtension=this.gl.getExtension("KHR_parallel_shader_compile"),1===(0,nN.env)().getNumber("WEBGL_VERSION")){const e="OES_texture_half_float";if(this.textureFloatExtension=gO(this.gl,"OES_texture_float"),xr(this.gl,e))this.textureHalfFloatExtension=gO(this.gl,e);else if((0,nN.env)().get("WEBGL_FORCE_F16_TEXTURES"))throw Error("GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.");if(this.colorBufferFloatExtension=this.gl.getExtension(r),xr(this.gl,n))this.colorBufferHalfFloatExtension=gO(this.gl,n);else if((0,nN.env)().get("WEBGL_FORCE_F16_TEXTURES"))throw Error("GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.")}else if(r="EXT_color_buffer_float",xr(this.gl,r))this.colorBufferFloatExtension=this.gl.getExtension(r);else if(xr(this.gl,n))this.colorBufferHalfFloatExtension=this.gl.getExtension(n);else throw Error("GL context does not support color renderable floats");this.vertexBuffer=xB(this.gl),this.indexBuffer=xW(this.gl),this.framebuffer=gq(this.gl),this.textureConfig=gA(this.gl,this.textureHalfFloatExtension)}get debug(){return(0,nN.env)().getBool("DEBUG")}dispose(){if(this.disposed)return;null!=this.program&&console.warn("Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing."),null!=this.outputTexture&&console.warn("Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.");let e=this.gl;gE(e,()=>e.finish()),gE(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,null)),gE(e,()=>e.deleteFramebuffer(this.framebuffer)),gE(e,()=>e.bindBuffer(e.ARRAY_BUFFER,null)),gE(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,null)),gE(e,()=>e.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(e,t){return this.throwIfDisposed(),xV(this.gl,e,t,this.textureConfig)}createFloat16MatrixTexture(e,t){return this.throwIfDisposed(),xq(this.gl,e,t,this.textureConfig)}createUnsignedBytesMatrixTexture(e,t){return this.throwIfDisposed(),xX(this.gl,e,t,this.textureConfig)}uploadPixelDataToTexture(e,t){this.throwIfDisposed(),x1(this.gl,e,t)}uploadDenseMatrixToTexture(e,t,r,n){this.throwIfDisposed(),x0(this.gl,e,t,r,n,this.textureConfig)}createFloat16PackedMatrixTexture(e,t){return this.throwIfDisposed(),xJ(this.gl,e,t,this.textureConfig)}createPackedMatrixTexture(e,t){return this.throwIfDisposed(),xY(this.gl,e,t,this.textureConfig)}deleteMatrixTexture(e){this.throwIfDisposed(),this.outputTexture===e&&(gQ(this.gl,this.framebuffer),this.outputTexture=null),gE(this.gl,()=>this.gl.deleteTexture(e))}downloadByteEncodedFloatMatrixFromOutputTexture(e,t,r){return this.downloadMatrixDriver(e,()=>x4(this.gl,t,r,this.textureConfig))}downloadPackedMatrixFromBuffer(e,t,r,n,a,s){return x5(this.gl,e,t,r,n,a,s,this.textureConfig)}downloadFloat32MatrixFromBuffer(e,t){return x3(this.gl,e,t)}createBufferFromTexture(e,t,r){this.bindTextureToFrameBuffer(e);let n=x2(this.gl,t,r,this.textureConfig);return this.unbindTextureToFrameBuffer(),n}createAndWaitForFence(){let e=this.createFence(this.gl);return this.pollFence(e)}createFence(e){let t,r;if((0,nN.env)().getBool("WEBGL_FENCE_API_ENABLED")){let n=e.fenceSync(e.SYNC_GPU_COMMANDS_COMPLETE,0);e.flush(),r=()=>{let t=e.clientWaitSync(n,0,0);return t===e.ALREADY_SIGNALED||t===e.CONDITION_SATISFIED},t=n}else(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0?(t=this.beginQuery(),this.endQuery(),r=()=>this.isQueryAvailable(t,(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))):r=()=>!0;return{query:t,isFencePassed:r}}downloadMatrixFromPackedTexture(e,t,r){return this.downloadMatrixDriver(e,()=>x6(this.gl,t,r))}createProgram(e){this.throwIfDisposed();let t=this.gl;null==this.vertexShader&&(this.vertexShader=xP(t));let r=gP(t);gE(t,()=>t.attachShader(r,this.vertexShader)),gE(t,()=>t.attachShader(r,e)),gB(t,r);let n=Object.assign(r,{vao:this.createVertexArray()});return this.debug&&gW(t,n),n}buildVao(e){this.setProgram(e),this.bindVertexArray(e.vao);let t=this.gl;gE(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,this.indexBuffer)),xQ(t,e,this.vertexBuffer)}deleteProgram(e){this.throwIfDisposed(),e===this.program&&(this.program=null),null!=e&&(gE(this.gl,()=>this.gl.deleteProgram(e)),this.deleteVertexArray(e.vao))}setProgram(e){this.throwIfDisposed(),this.program=e,null!=this.program&&this.debug&&gW(this.gl,this.program),gE(this.gl,()=>this.gl.useProgram(e))}getUniformLocation(e,t,r=!0){return(this.throwIfDisposed(),r)?gK(this.gl,e,t):gY(this.gl,e,t)}getAttributeLocation(e,t){return this.throwIfDisposed(),gE(this.gl,()=>this.gl.getAttribLocation(e,t))}getUniformLocationNoThrow(e,t){return this.throwIfDisposed(),this.gl.getUniformLocation(e,t)}setInputMatrixTexture(e,t,r){this.throwIfDisposed(),this.throwIfNoProgram(),gZ(this.gl,e,t,r)}setOutputMatrixTexture(e,t,r){this.setOutputMatrixTextureDriver(e,r,t)}setOutputPackedMatrixTexture(e,t,r){this.throwIfDisposed();let[n,a]=gR(t,r);this.setOutputMatrixTextureDriver(e,n,a)}setOutputMatrixWriteRegion(e,t,r,n){this.setOutputMatrixWriteRegionDriver(r,e,n,t)}setOutputPackedMatrixWriteRegion(e,t,r,n){throw Error("setOutputPackedMatrixWriteRegion not implemented.")}debugValidate(){null!=this.program&&gW(this.gl,this.program),g0(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();let e=this.gl;this.debug&&(console.assert(this.getVertexArray()===this.program.vao,"VAO changed between setProgram and executeProgram!"),this.debugValidate()),gE(e,()=>e.drawElements(e.TRIANGLES,6,e.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),gE(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return null==this.disjointQueryTimerExtension&&(this.disjointQueryTimerExtension=gO(this.gl,2===(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")?"EXT_disjoint_timer_query_webgl2":"EXT_disjoint_timer_query")),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(2===(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")){let e=this.gl,t=this.getQueryTimerExtensionWebGL2(),r=e.createQuery();return e.beginQuery(t.TIME_ELAPSED_EXT,r),r}let e=this.getQueryTimerExtensionWebGL1(),t=e.createQueryEXT();return e.beginQueryEXT(e.TIME_ELAPSED_EXT,t),t}endQuery(){if(2===(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")){let e=this.gl,t=this.getQueryTimerExtensionWebGL2();e.endQuery(t.TIME_ELAPSED_EXT);return}let e=this.getQueryTimerExtensionWebGL1();e.endQueryEXT(e.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(e){return await nR.util.repeatedTry(()=>this.disposed||this.isQueryAvailable(e,(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))),this.getQueryTime(e,(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))}getQueryTime(e,t){if(0===t)return null;if(2===t){let t=this.gl;return t.getQueryParameter(e,t.QUERY_RESULT)/1e6}{let t=this.getQueryTimerExtensionWebGL1();return t.getQueryObjectEXT(e,t.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(e,t){if(0===t)return!0;if(2===t){let t=this.gl,r=this.getQueryTimerExtensionWebGL2(),n=t.getQueryParameter(e,t.QUERY_RESULT_AVAILABLE);return null==this.disjoint&&(this.disjoint=this.gl.getParameter(r.GPU_DISJOINT_EXT)),n&&!this.disjoint}{let t=this.getQueryTimerExtensionWebGL1(),r=t.getQueryObjectEXT(e,t.QUERY_RESULT_AVAILABLE_EXT);return null==this.disjoint&&(this.disjoint=this.gl.getParameter(t.GPU_DISJOINT_EXT)),r&&!this.disjoint}}pollFence(e){return new Promise(t=>{this.addItemToPoll(()=>e.isFencePassed(),()=>t())})}pollItems(){let e=function(e){let t=0;for(;t<e.length&&e[t]();++t);return t-1}(this.itemsToPoll.map(e=>e.isDoneFn));for(let t=0;t<=e;++t){let{resolveFn:e}=this.itemsToPoll[t];e()}this.itemsToPoll=this.itemsToPoll.slice(e+1)}addItemToPoll(e,t){let r;this.itemsToPoll.push({isDoneFn:e,resolveFn:t}),this.itemsToPoll.length>1||("setTimeoutCustom"in(0,nN.env)().platform&&(r=(0,nN.env)().platform.setTimeoutCustom.bind((0,nN.env)().platform)),nR.util.repeatedTry(()=>(this.pollItems(),0===this.itemsToPoll.length),()=>0,null,r))}bindTextureToFrameBuffer(e){this.throwIfDisposed(),gJ(this.gl,e,this.framebuffer),this.debug&&g0(this.gl)}unbindTextureToFrameBuffer(){null!=this.outputTexture?(gJ(this.gl,this.outputTexture,this.framebuffer),this.debug&&g0(this.gl)):gQ(this.gl,this.framebuffer)}downloadMatrixDriver(e,t){this.bindTextureToFrameBuffer(e);let r=t();return this.unbindTextureToFrameBuffer(),r}setOutputMatrixTextureDriver(e,t,r){this.throwIfDisposed();let n=this.gl;gJ(n,e,this.framebuffer),this.debug&&g0(n),this.outputTexture=e,gE(n,()=>n.viewport(0,0,t,r)),gE(n,()=>n.scissor(0,0,t,r))}setOutputMatrixWriteRegionDriver(e,t,r,n){this.throwIfDisposed(),gE(this.gl,()=>this.gl.scissor(e,t,r,n))}throwIfDisposed(){if(this.disposed)throw Error("Attempted to use disposed GPGPUContext.")}throwIfNoProgram(){if(null==this.program)throw Error("No GPU program is currently set.")}}let{addImpl:x7,bincountImpl:x9,bincountReduceImpl:ye,bitwiseAndImpl:yt,castImpl:yr,ceilImpl:yn,concatImpl:ya,equalImpl:ys,expImpl:yi,expm1Impl:yo,floorImpl:yl,gatherNdImpl:yu,gatherV2Impl:yh,greaterImpl:yd,greaterEqualImpl:yp,lessImpl:yc,lessEqualImpl:yf,linSpaceImpl:ym,logImpl:yg,maxImpl:yx,maximumImpl:yy,minimumImpl:yb,multiplyImpl:yv,negImpl:yw,notEqualImpl:yC,prodImpl:yI,raggedGatherImpl:yk,raggedRangeImpl:yS,raggedTensorToTensorImpl:yT,rangeImpl:yN,rsqrtImpl:y$,scatterImpl:yR,sigmoidImpl:yA,simpleAbsImpl:yE,sliceImpl:yF,sparseFillEmptyRowsImpl:yD,sparseReshapeImpl:yO,sparseSegmentReductionImpl:yL,sqrtImpl:yz,staticRegexReplaceImpl:y_,stridedSliceImpl:yM,stringNGramsImpl:yP,stringSplitImpl:yB,stringToHashBucketFastImpl:yW,subImpl:yG,tileImpl:yU,topKImpl:yV,transposeImpl:yH,uniqueImpl:yq}=cs;function yj(e,t){return["x","y","z","w","u","v"].slice(0,t).map(t=>`${e}.${t}`)}function yX(e,t){return 1===t?[e]:yj(e,t)}class yK{constructor(e){if(this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.enableShapeUniforms=xE(this.outputShape.length),0===this.rank)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{const e=yX("rc",this.rank),t=xS(this.rank),r=this.getOutOfBoundsCondition(e),n=this.getSetup(e),a=this.getOutput(e);this.userCode=`
        void main() {
          ${t} rc = getOutputCoords();

          if(${r}) {
            setOutput(vec4(0));
          } else {
            ${n}

            setOutput(vec4(${a}));
          }
        }
      `}}getSourceCoordsArr(e){let t=[];for(let r=0;r<=1;r++)for(let n=0;n<=1;n++){let a=`${0===r?"r":"rp1"}, ${0===n?"c":"cp1"}`;for(let t=2;t<this.rank;t++)a=`${e[e.length-1-t]},`+a;t.push(a)}return t}getOutOfBoundsCondition(e){if(1===this.rank)return`rc > ${this.enableShapeUniforms?"outShape":this.outputShape[0]}`;let t="";for(let r=this.rank-2;r<this.rank;r++)t+=`${e[r]} >= ${this.enableShapeUniforms?`outShape[${r}]`:this.outputShape[r]}`,r<this.rank-1&&(t+="||");return t}getSetup(e){if(1===this.rank)return"";let t=e.slice(-2),r=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],n=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${t[0]};
      int c = ${t[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${r};
      bool rEdge = rp1 >= ${n};
    `}getOutput(e){let t=this.getSourceCoordsArr(e);if(1===this.rank){let e=this.enableShapeUniforms?"outShape":this.outputShape[0];return`getA(rc), (rc + 1 >= ${e} ? 0. : getA(rc + 1)), 0, 0`}return`getA(${t[0]}),
            cEdge ? 0. : getA(${t[1]}),
            rEdge ? 0. : getA(${t[2]}),
            rEdge || cEdge ? 0. : getA(${t[3]})`}}class yY{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec3"}],this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length);let r="";for(let e=0;e<4;e++){let t="thisRC = rc;";e%2==1&&(t+="thisRC.z += 1;"),e>1&&(t+="thisRC.y += 1;"),r+=`
        ${t}
        ${e>0?"if(thisRC.y < rows && thisRC.z < cols){":""}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${e}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${e>0?"}":""}
      `}this.userCode=`
      ${function(e,t){let r=t?function(e,t,r="index"){let n=function(e,t){let r=e.length,n=e.map(e=>`${t}[${e}]`),a=Array(r-1);a[r-2]=n[r-1];for(let e=r-3;e>=0;--e)a[e]=`(${a[e+1]} * ${n[e+1]})`;return a}(e.map((e,t)=>t),t);return n.map((t,a)=>{let s=`int ${e[a]} = ${r} / ${n[a]}`,i=a===n.length-1?`int ${e[a+1]} = ${r} - ${e[a]} * ${n[a]}`:`index -= ${e[a]} * ${n[a]}`;return`${s}; ${i};`}).join("")}(["r","c","d"],"inputShape"):xp(["r","c","d"],e);return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${r}
      return ivec3(r, c, d);
    }
  `}(t,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?xm():xf(e)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?"outShape[1]":e[1]};
        int cols = ${this.enableShapeUniforms?"outShape[2]":e[2]};

        ${r}

        setOutput(result);
      }
    `}}class yZ{constructor(e){this.gpgpu=e,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.usedTextures={},this.logEnabled=!1}acquireTexture(e,t,r){let n,a=yQ(t,r),s=y0(e,a,r);s in this.freeTextures||(this.freeTextures[s]=[]),s in this.usedTextures||(this.usedTextures[s]=[]);let i=yJ(e,a,this.gpgpu.gl,this.gpgpu.textureConfig,r);if(this.freeTextures[s].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=i,this.log();let e=this.freeTextures[s].pop();return this.usedTextures[s].push(e),e}return a===m.PACKED_2X2_FLOAT32?n=this.gpgpu.createPackedMatrixTexture(e[0],e[1]):a===m.PACKED_2X2_FLOAT16?n=this.gpgpu.createFloat16PackedMatrixTexture(e[0],e[1]):a===m.UNPACKED_FLOAT32?n=this.gpgpu.createFloat32MatrixTexture(e[0],e[1]):a===m.UNPACKED_FLOAT16?n=this.gpgpu.createFloat16MatrixTexture(e[0],e[1]):a===m.PACKED_4X1_UNSIGNED_BYTE&&(n=this.gpgpu.createUnsignedBytesMatrixTexture(e[0],e[1])),this.usedTextures[s].push(n),this.numUsedTextures++,this._numBytesAllocated+=i,this.log(),n}releaseTexture(e,t,r,n){if(null==this.freeTextures)return;let a=yQ(r,n),s=y0(t,a,n);s in this.freeTextures||(this.freeTextures[s]=[]);let i=yJ(t,a,this.gpgpu.gl,this.gpgpu.textureConfig,n),o=(0,nN.env)().getNumber("WEBGL_DELETE_TEXTURE_THRESHOLD");-1!==o&&this._numBytesAllocated>o?(this.gpgpu.deleteMatrixTexture(e.texture),this._numBytesAllocated-=i):(this.freeTextures[s].push(e),this.numFreeTextures++,this._numBytesFree+=i),this.numUsedTextures--;let l=this.usedTextures[s],u=l&&l.indexOf(e);if(null==u||u<0)throw Error("Cannot release a texture that was never provided by this texture manager");l[u]=l[l.length-1],l.pop(),this.log()}log(){if(!this.logEnabled)return;let e=this.numFreeTextures+this.numUsedTextures;console.log("Free/Used",`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${e})`);let t=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*t)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(null!=this.freeTextures){for(let e in this.freeTextures)this.freeTextures[e].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});for(let e in this.usedTextures)this.usedTextures[e].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}}function yJ(e,t,r,n,a){let s,i=function(e,t){switch(e){case m.PACKED_2X2_FLOAT32:return xK(t);case m.PACKED_2X2_FLOAT16:return xZ(t);case m.UNPACKED_FLOAT32:return xU(t);case m.UNPACKED_FLOAT16:return xH(t);case m.PACKED_4X1_UNSIGNED_BYTE:return xj(t);default:throw Error(`Unknown physical texture type ${e}`)}}(t,n);if(a){let[t,r]=gR(e[0],e[1]);s=t*r}else{var o;let[t,r]=(o=e[0],[e[1],o]);s=t*r}return s*function(e,t){if(t===e.R32F)return 4;if(t===e.R16F)return 2;if(t===e.RGBA32F)return 16;if(t===e.RGBA)return 16;if(t===e.RGBA16F)return 8;else if(t===e.RGBA8)return 4;throw Error(`Unknown internal format ${t}`)}(r,i)}function yQ(e,t){if(e===f.UPLOAD)return m.PACKED_2X2_FLOAT32;if(e===f.RENDER||null==e)return(0,nN.env)().getBool("WEBGL_RENDER_FLOAT32_ENABLED")?t?m.PACKED_2X2_FLOAT32:m.UNPACKED_FLOAT32:t?m.PACKED_2X2_FLOAT16:m.UNPACKED_FLOAT16;if(e===f.DOWNLOAD||e===f.PIXELS)return m.PACKED_4X1_UNSIGNED_BYTE;throw Error(`Unknown logical texture type ${e}`)}function y0(e,t,r){return`${e[0]}_${e[1]}_${t}_${r}`}class y1{constructor(e,t){this.variableNames=["A"],this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${t}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}}let y2="if (isnan(x)) return x;",y3="return abs(x);",y4=y2+`
  return (x < 0.0) ? 0.0 : x;
`,y5=y2+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,y6="return x;",y8=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,y7=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,y9=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;class be{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${t}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}}class bt{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length);const t=e.length,r=yX("rc",t),n=xS(t),a=function(e,t){if(1===e)return"rc";let r="";for(let n=0;n<e;n++)r+=t[n],n<e-1&&(r+=",");return r}(t,r),s=r.slice(-2),i=t<=1?"rc":`vec2(${s.join(",")})`;this.userCode=`
      void main() {
        ${n} rc = getOutputCoords();
        vec4 packedInput = getA(${a});

        setOutput(getChannel(packedInput, ${i}));
      }
    `}}let br=h8.whereImpl,bn={},ba=(0,nN.env)().getNumber("CPU_HANDOFF_SIZE_THRESHOLD");class bs extends xh.KernelBackend{nextDataId(){return bs.nextDataId++}constructor(e){let t;if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!(0,nN.env)().getBool("HAS_WEBGL"))throw Error("WebGL is not supported on this device");null!=e?(t=e instanceof x8?e:new x8(gN((0,nN.env)().getNumber("WEBGL_VERSION"),e)),this.binaryCache={},this.gpgpuCreatedLocally=!1):(t=new x8(gN((0,nN.env)().getNumber("WEBGL_VERSION"))),this.binaryCache=function(e){return e in bn||(bn[e]={}),bn[e]}((0,nN.env)().getNumber("WEBGL_VERSION")),this.gpgpuCreatedLocally=!0),this.gpgpu=t,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new yZ(this.gpgpu),this.numMBBeforeWarning=null==(0,nN.env)().global.screen?1024:(0,nN.env)().global.screen.height*(0,nN.env)().global.screen.width*window.devicePixelRatio*600/1024/1024,this.texData=new xh.DataStorage(this,(0,n$.engine)())}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}writeTexture(e,t,r,n,a,s){let i=this.makeTensorInfo(t,r),o=this.texData.get(i.dataId);o.isPacked=!1,o.texture={texture:e,texShape:[n,a]},o.texShape=[n,a];let l=new x_(g6(t),!1,s),u=this.runWebGLProgram(l,[i],r,[[n,a]]);return u.shape=t,o.texture=null,this.disposeIntermediateTensorInfo(i),u.dataId}write(e,t,r){if(((0,nN.env)().getBool("WEBGL_CHECK_NUMERICAL_PROBLEMS")||(0,nN.env)().getBool("DEBUG"))&&this.checkNumericalProblems(e),"complex64"===r&&null!=e)throw Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");let n={id:this.nextDataId()};return this.texData.set(n,{shape:t,dtype:r,values:e,usage:f.UPLOAD,refCount:1}),n}refCount(e){return this.texData.has(e)?this.texData.get(e).refCount:0}incRef(e){let t=this.texData.get(e);t.refCount++}decRef(e){if(this.texData.has(e)){let t=this.texData.get(e);t.refCount--}}move(e,t,r,n,a){if((0,nN.env)().getBool("DEBUG")&&this.checkNumericalProblems(t),"complex64"===n)throw Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.texData.set(e,{shape:r,dtype:n,values:t,usage:f.UPLOAD,refCount:a})}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}readSync(e){let t,r,{values:n,dtype:a,complexTensorInfos:s,slice:i,shape:o,isPacked:l}=this.texData.get(e);if(null!=i){let t;t=l?new be(o,y6):new y1(o,y6);let r=this.runWebGLProgram(t,[{dataId:e,shape:o,dtype:a}],a),n=this.readSync(r.dataId);return this.disposeIntermediateTensorInfo(r),n}if(null!=n)return this.convertAndCacheOnCPU(e);if("string"===a)return n;let u=null!=this.activeTimers;if(u&&(t=nR.util.now()),"complex64"===a){let e=this.readSync(s.real.dataId),t=this.readSync(s.imag.dataId);r=tN.mergeRealAndImagArrays(e,t)}else r=this.getValuesFromTexture(e);return u&&(this.downloadWaitMs+=nR.util.now()-t),this.convertAndCacheOnCPU(e,r)}async read(e){let t,r;if(this.pendingRead.has(e)){let t=this.pendingRead.get(e);return new Promise(e=>t.push(e))}let{values:n,shape:a,slice:s,dtype:i,complexTensorInfos:o,isPacked:l}=this.texData.get(e);if(null!=s){let t;t=l?new be(a,y6):new y1(a,y6);let r=this.runWebGLProgram(t,[{dataId:e,shape:a,dtype:i}],i),n=this.read(r.dataId);return this.disposeIntermediateTensorInfo(r),n}if(null!=n)return this.convertAndCacheOnCPU(e);if((0,nN.env)().getBool("DEBUG")&&!(0,nN.env)().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")&&2===(0,nN.env)().getNumber("WEBGL_VERSION"))throw Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.");let u=null;if("complex64"!==i&&(0,nN.env)().get("WEBGL_BUFFER_SUPPORTED")){t=this.decode(e);let r=this.texData.get(t.dataId);u=this.gpgpu.createBufferFromTexture(r.texture.texture,...g$(a))}if(this.pendingRead.set(e,[]),"complex64"!==i&&await this.gpgpu.createAndWaitForFence(),"complex64"===i){let e=await Promise.all([this.read(o.real.dataId),this.read(o.imag.dataId)]),t=e[0],n=e[1];r=tN.mergeRealAndImagArrays(t,n)}else if(null==u)r=this.getValuesFromTexture(e);else{let e=nR.util.sizeFromShape(a);r=this.gpgpu.downloadFloat32MatrixFromBuffer(u,e)}if(null!=t&&this.disposeIntermediateTensorInfo(t),null!=u){let e=this.gpgpu.gl;gE(e,()=>e.deleteBuffer(u))}let h=this.convertAndCacheOnCPU(e,r),d=this.pendingRead.get(e);return this.pendingRead.delete(e),d.forEach(e=>e(h)),this.pendingDisposal.has(e)&&(this.pendingDisposal.delete(e),this.disposeData(e)&&(0,n$.engine)().removeDataId(e,this),this.pendingDeletes--),h}readToGPU(e,t={}){let{values:r,shape:n,slice:a,dtype:s,isPacked:i,texture:o}=this.texData.get(e);if("complex64"===s)throw Error("Does not support reading texture for complex64 dtype.");if(null!=a){let r;r=i?new be(n,y6):new y1(n,y6);let a=this.runWebGLProgram(r,[{dataId:e,shape:n,dtype:s}],s),o=this.readToGPU(a,t);return this.disposeIntermediateTensorInfo(a),o}if(null==o)if(null!=r)throw Error("Data is not on GPU but on CPU.");else throw Error("There is no data on GPU or CPU.");let l=this.decode(e,t.customTexShape);return Object.assign({tensorRef:(0,n$.engine)().makeTensorFromTensorInfo(l)},this.texData.get(l.dataId).texture)}bufferSync(e){let t=this.readSync(e.dataId);if("string"===e.dtype)try{let r=t.map(e=>nR.util.decodeString(e));return(0,h5.buffer)(e.shape,e.dtype,r)}catch(e){throw Error("Failed to decode encoded string bytes into utf-8")}return(0,h5.buffer)(e.shape,e.dtype,t)}checkNumericalProblems(e){if(null!=e)for(let t=0;t<e.length;t++){let r=e[t];if(!gF(r)){if((0,nN.env)().getBool("WEBGL_RENDER_FLOAT32_CAPABLE"))throw Error(`The value ${r} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`);throw Error(`The value ${r} cannot be represented on this device.`)}}}getValuesFromTexture(e){let{shape:t,dtype:r,isPacked:n}=this.texData.get(e),a=nR.util.sizeFromShape(t);if((0,nN.env)().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")){let r=this.decode(e),n=this.texData.get(r.dataId),s=this.gpgpu.downloadMatrixFromPackedTexture(n.texture.texture,...g$(t)).subarray(0,a);return this.disposeIntermediateTensorInfo(r),s}let s=(0,nN.env)().getBool("WEBGL_PACK")&&!0===n,i=s?g6(t):t,o=s?new xL(i):new xO(i),l=this.runWebGLProgram(o,[{shape:i,dtype:r,dataId:e}],"float32"),u=this.texData.get(l.dataId),h=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(u.texture.texture,u.texShape[0],u.texShape[1]).subarray(0,a);return this.disposeIntermediateTensorInfo(l),h}timerAvailable(){return(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0}time(e){let t=this.activeTimers,r=[],n=!1;null==this.programTimersStack?(this.programTimersStack=r,n=!0):this.activeTimers.push(r),this.activeTimers=r,e();let a=nR.util.flatten(this.activeTimers.map(e=>e.query)).filter(e=>null!=e),s=nR.util.flatten(this.activeTimers.map(e=>e.name)).filter(e=>null!=e);this.activeTimers=t,n&&(this.programTimersStack=null);let i={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if((0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0){let e=await Promise.all(a);i.kernelMs=nR.util.sum(e),i.getExtraProfileInfo=()=>e.map((e,t)=>({name:s[t],ms:e})).map(e=>`${e.name}: ${e.ms}`).join(", ")}else i.kernelMs={error:"WebGL query timers are not supported in this environment."};return this.uploadWaitMs=0,this.downloadWaitMs=0,i})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.beginQuery():{startMs:nR.util.now(),endMs:null}}endTimer(e){return(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.endQuery():e.endMs=nR.util.now(),e}async getQueryTime(e){return(0,nN.env)().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.waitForQueryAndGetTime(e):e.endMs-e.startMs}disposeData(e,t=!1){if(this.pendingDisposal.has(e))return!1;if(!this.texData.has(e))return!0;if(t?this.texData.get(e).refCount=0:this.texData.get(e).refCount--,!t&&this.texData.get(e).refCount>0)return!1;if(this.pendingRead.has(e))return this.pendingDisposal.add(e),this.pendingDeletes++,!1;this.releaseGPUData(e);let{complexTensorInfos:r}=this.texData.get(e);return null!=r&&(this.disposeData(r.real.dataId,t),this.disposeData(r.imag.dataId,t)),this.texData.delete(e),!0}releaseGPUData(e){let{texture:t,dtype:r,texShape:n,usage:a,isPacked:s,slice:i}=this.texData.get(e),o=i&&i.origDataId||e,l=this.dataRefCount.get(o);l>1?this.dataRefCount.set(o,l-1):(this.dataRefCount.delete(o),null!=t&&(this.numBytesInGPU-=this.computeBytes(n,r),this.textureManager.releaseTexture(t,n,a,s)));let u=this.texData.get(e);u.texture=null,u.texShape=null,u.isPacked=!1,u.slice=null}getTexture(e){return this.uploadToGPU(e),this.texData.get(e).texture.texture}getDataInfo(e){return this.texData.get(e)}shouldExecuteOnCPU(e,t=ba){return(0,nN.env)().getBool("WEBGL_CPU_FORWARD")&&e.every(e=>null==this.texData.get(e.dataId).texture&&nR.util.sizeFromShape(e.shape)<t)}getGPGPUContext(){return this.gpgpu}where(e){tN.warn("tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead");let t=e.dataSync();return br(e.shape,t)}packedUnaryOp(e,t,r){let n=new be(e.shape,t),a=this.compileAndRun(n,[e],r);return(0,n$.engine)().makeTensorFromTensorInfo(a)}abs(e){if(this.shouldExecuteOnCPU([e])&&"complex64"!==e.dtype){let t=yE(this.texData.get(e.dataId).values);return this.makeOutput(e.shape,e.dtype,t)}if((0,nN.env)().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(e,y3,e.dtype);let t=new y1(e.shape,y3),r=this.compileAndRun(t,[e]);return(0,n$.engine)().makeTensorFromTensorInfo(r)}makeTensorInfo(e,t,r){let n;if("string"===t&&null!=r&&r.length>0&&nR.util.isString(r[0])){let a=r.map(e=>nR.util.encodeString(e));n=this.write(a,e,t)}else n=this.write(r,e,t);return this.texData.get(n).usage=null,{dataId:n,shape:e,dtype:t}}makeOutput(e,t,r){return(0,n$.engine)().makeTensorFromTensorInfo(this.makeTensorInfo(e,t,r),this)}unpackTensor(e){let t=new bt(e.shape);return this.runWebGLProgram(t,[e],e.dtype)}packTensor(e){let t=new yK(e.shape);return this.runWebGLProgram(t,[e],e.dtype,null,!0)}packedReshape(e,t){let r=[g4(e.shape),...g5(e.shape)],n={dtype:e.dtype,shape:r,dataId:e.dataId},a=new yY([g4(t),...g5(t)],r),s=this.runWebGLProgram(a,[n],e.dtype,[r],!0);return{dataId:s.dataId,shape:t,dtype:s.dtype}}decode(e,t){let r,{isPacked:n,shape:a,dtype:s}=this.texData.get(e);if(null!=t){let e=nR.util.sizeFromShape(a),r=t[0]*t[1]*4;nR.util.assert(e<=r,()=>"customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.")}let i=g6(a);r=n?new xD(i):new xF(i);let o=[null!=t?t:g$(i)],l=this.runWebGLProgram(r,[{shape:i,dtype:s,dataId:e}],s,o,!0,t);return{dtype:s,shape:a,dataId:l.dataId}}runWebGLProgram(e,t,r,n,a=!1,s){let i,o,l,u=this.makeTensorInfo(e.outputShape,r),h=this.texData.get(u.dataId);if(e.packedOutput&&(h.isPacked=!0),e.outPackingScheme===c.DENSE&&(h.texShape=(null!=s?s:g$(e.outputShape)).map(e=>2*e)),null!=e.outTexUsage&&(h.usage=e.outTexUsage),0===nR.util.sizeFromShape(u.shape))return h.values=nR.util.getTypedArrayFromDType(u.dtype,0),u;let d=[],p=t.map(t=>{if("complex64"===t.dtype)throw Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");let r=this.texData.get(t.dataId);if(null==r.texture){if(!e.packedInputs&&nR.util.sizeFromShape(t.shape)<=(0,nN.env)().getNumber("WEBGL_SIZE_UPLOAD_UNIFORM"))return{shape:t.shape,texData:null,isUniform:!0,uniformValues:r.values};e.packedInputs&&(r.isPacked=!0,r.shape=t.shape)}if(this.uploadToGPU(t.dataId),!!r.isPacked!=!!e.packedInputs)t=r.isPacked?this.unpackTensor(t):this.packTensor(t),d.push(t),r=this.texData.get(t.dataId);else if(r.isPacked&&!g7(r.shape,t.shape)){let e=t,n=t.shape;t.shape=r.shape,t=this.packedReshape(t,n),d.push(t),r=this.texData.get(t.dataId),e.shape=n}return{shape:t.shape,texData:r,isUniform:!1}});this.uploadToGPU(u.dataId);let f={shape:u.shape,texData:h,isUniform:!1},m=(o="",p.concat(f).forEach(t=>{let r=null!=t.texData&&null!=t.texData.slice&&t.texData.slice.flatOffset>0;if(e.enableShapeUniforms&&!t.isUniform){let n=t.texData.texShape,{useSqueezeShape:a,uniformShape:s,keptDims:i}=xT(e.packedInputs,t.shape,n),l="",u="",h="";if(1===s.length&&e.packedInputs){let e=[Math.ceil(n[0]/2),Math.ceil(n[1]/2)];l=`${e[0]>1}_${e[1]>1}`}else if(2!==s.length||e.packedInputs){if(s.length>2&&!e.packedInputs){let e=nR.util.computeStrides(s);h=`${e[0]===n[1]}_${e[e.length-1]===n[1]}`}}else u=`${s[0]>1}_${s[1]>1}`;let d=t.shape.length,p=2===s.length&&nR.util.arraysEqual(t.shape,n),c=1===nR.util.sizeFromShape(t.shape),m=tN.getBroadcastDims(t.shape,f.shape),g=!e.packedInputs&&d===f.shape.length&&nR.util.arraysEqual(n,f.texData.texShape),x=e.packedInputs||s.length>2?"":`${n[0]>1}_${n[1]>1}`;o+=`${d}_${g}_${a?i:""}_${s.length}_${c}_${m}_${p}_${l}_${u}_${h}_${x}_${r}`}else{let e=t.isUniform?"uniform":t.texData.texShape;o+=`${t.shape}_${e}_${r}`}}),l=e.userCode,e.constructor.name+("_"+o+"_"+l)+`${(0,nN.env)().getNumber("WEBGL_VERSION")}`),g=this.getAndSaveBinary(m,()=>{var t;let r,n,a,s,i,o;return t=this.gpgpu,n=(r=p.map((t,r)=>{let n={logicalShape:t.shape,texShape:t.isUniform?null:t.texData.texShape,isUniform:t.isUniform,isPacked:!t.isUniform&&t.texData.isPacked,flatOffset:null};return null!=t.texData&&null!=t.texData.slice&&t.texData.slice.flatOffset>0&&(n.flatOffset=t.texData.slice.flatOffset),{name:e.variableNames[r],shapeInfo:n}})).map(e=>e.shapeInfo),s=function(e,t,r){var n,a,s,i;let o,l,u=[];if(e.forEach(e=>{let t=nR.util.sizeFromShape(e.shapeInfo.logicalShape);if(e.shapeInfo.isUniform?u.push(`uniform float ${e.name}${t>1?`[${t}]`:""};`):(u.push(`uniform sampler2D ${e.name};`),u.push(`uniform int offset${e.name};`)),r.enableShapeUniforms){let{uniformShape:t}=xT(r.packedInputs,e.shapeInfo.logicalShape,e.shapeInfo.texShape);switch(t.length){case 1:u.push(`uniform int ${e.name}Shape;`);break;case 2:u.push(`uniform ivec2 ${e.name}Shape;`);break;case 3:u.push(`uniform ivec3 ${e.name}Shape;`);break;case 4:u.push(`uniform ivec4 ${e.name}Shape;`)}u.push(`uniform ivec2 ${e.name}TexShape;`)}}),r.enableShapeUniforms){switch(t.logicalShape.length){case 1:u.push("uniform int outShape;");break;case 2:u.push("uniform ivec2 outShape;"),u.push("uniform int outShapeStrides;");break;case 3:u.push("uniform ivec3 outShape;"),u.push("uniform ivec2 outShapeStrides;");break;case 4:u.push("uniform ivec4 outShape;"),u.push("uniform ivec3 outShapeStrides;")}u.push("uniform ivec2 outTexShape;")}r.customUniforms&&r.customUniforms.forEach(e=>{u.push(`uniform ${e.type} ${e.name}${e.arrayIndex?`[${e.arrayIndex}]`:""};`)});let h=u.join("\n"),d=e.map(e=>(function(e,t,r=!1,n){let a="";r?a+=function e(t,r){switch(t.shapeInfo.logicalShape.length){case 0:let n,a,s;return a="get"+(n=t.name).charAt(0).toUpperCase()+n.slice(1),s=xd(),`
    vec4 ${a}() {
      return ${s.texture2D}(${n}, halfCR);
    }
  `;case 1:return function(e,t){let r=e.name,n="get"+r.charAt(0).toUpperCase()+r.slice(1),a=e.shapeInfo.texShape,s=xd();if(t)return`
    vec4 ${n}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${r}TexShape[0]) / 2.0), ceil(float(${r}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${s.texture2D}(${r}, uv);
    }
  `;let i=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)];return`
    vec4 ${n}(int index) {
      vec2 uv = packedUVfrom1D(
        ${i[0]}, ${i[1]}, index);
      return ${s.texture2D}(${r}, uv);
    }
  `}(t,r);case 2:return function(e,t){let r=e.shapeInfo.logicalShape,n=e.name,a="get"+n.charAt(0).toUpperCase()+n.slice(1),s=e.shapeInfo.texShape,i=s[0],o=s[1],l=xd();if(null!=s&&nR.util.arraysEqual(r,s))return t?`
      vec4 ${a}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${n}TexShape[1], ${n}TexShape[0]);

        return ${l.texture2D}(${n}, uv);
      }
    `:`
      vec4 ${a}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${o}.0, ${i}.0);

        return ${l.texture2D}(${n}, uv);
      }
    `;if(t)return`
    vec4 ${a}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${n}TexShape[0]) / 2.0), ceil(float(${n}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${n}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${l.texture2D}(${n}, uv);
    }
  `;let u=[Math.ceil(s[0]/2),Math.ceil(s[1]/2)],h=Math.ceil(r[1]/2);return`
    vec4 ${a}(int row, int col) {
      vec2 uv = packedUVfrom2D(${h}, ${u[0]}, ${u[1]}, row, col);
      return ${l.texture2D}(${n}, uv);
    }
  `}(t,r);case 3:return function(t,r){let n=t.shapeInfo.logicalShape,a=t.name,s="get"+a.charAt(0).toUpperCase()+a.slice(1),i=t.shapeInfo.texShape,o=[Math.ceil(i[0]/2),Math.ceil(i[1]/2)];if(1===n[0]){let a=xN(t,n.slice(1));return`
        ${e(a,r)}
        vec4 ${s}(int b, int row, int col) {
          return ${s}(${x$(["b","row","col"],[1,2])});
        }
      `}let l=xd();if(r)return`
    vec4 ${s}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${a}TexShape[0]) / 2.0), ceil(float(${a}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${a}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${a}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${l.texture2D}(${a}, uv);
    }
  `;let u=o[0],h=o[1],d=Math.ceil(n[2]/2),p=d*Math.ceil(n[1]/2);return`
    vec4 ${s}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${u}, ${h}, ${p}, ${d}, b, row, col);
      return ${l.texture2D}(${a}, uv);
    }
  `}(t,r);default:return function(e,t){let r=e.name,n="get"+r.charAt(0).toUpperCase()+r.slice(1),a=xd();if(t)return`
    vec4 ${n}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${r}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${r}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${r}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${r}TexShape[0]) / 2.0), ceil(float(${r}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${a.texture2D}(${r}, uv);
    }
  `;let s=e.shapeInfo.logicalShape,i=s.length,o=e.shapeInfo.texShape,l=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)],u=l[0],h=l[1],d=Math.ceil(s[i-1]/2),p=d*Math.ceil(s[i-2]/2),c="int b, int row, int col",f=`b * ${p} + (row / 2) * ${d} + (col / 2)`;for(let e=2;e<i-1;e++)c=`int b${e}, `+c,p*=s[i-e-1],f=`b${e} * ${p} + `+f;return`
    vec4 ${n}(${c}) {
      int index = ${f};
      int texR = index / ${h};
      int texC = index - texR * ${h};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${h}, ${u});
      return ${a.texture2D}(${r}, uv);
    }
  `}(t,r)}}(e,n):a+=function e(t,r=!1){let n=t.shapeInfo.logicalShape;switch(n.length){case 0:return function(e,t){let r=e.name,n="get"+r.charAt(0).toUpperCase()+r.slice(1);if(e.shapeInfo.isUniform)return`float ${n}() {return ${r};}`;let[a,s]=e.shapeInfo.texShape;if(1===a&&1===s)return`
      float ${n}() {
        return sampleTexture(${r}, halfCR);
      }
    `;let i=xI(r);if(t)return`
    float ${n}() {
      vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], ${i});
      return sampleTexture(${r}, uv);
    }
  `;let[o,l]=e.shapeInfo.texShape;return`
    float ${n}() {
      vec2 uv = uvFromFlat(${o}, ${l}, ${i});
      return sampleTexture(${r}, uv);
    }
  `}(t,r);case 1:return function(e,t){let r=e.name,n="get"+r.charAt(0).toUpperCase()+r.slice(1);if(e.shapeInfo.isUniform)return`
      float ${n}(int index) {
        ${xk(e)}
      }
    `;let a=e.shapeInfo.texShape,s=a[0],i=a[1];if(1===i&&1===s)return`
      float ${n}(int index) {
        return sampleTexture(${r}, halfCR);
      }
    `;let o=xI(r);return 1===i?t?`
      float ${n}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${o}) + 0.5) / float(${r}TexShape[0]));
        return sampleTexture(${r}, uv);
      }
    `:`
      float ${n}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${o}) + 0.5) / ${s}.0);
        return sampleTexture(${r}, uv);
      }
    `:1===s?t?`
      float ${n}(int index) {
        vec2 uv = vec2((float(index + ${o}) + 0.5) / float(${r}TexShape[1]), 0.5);
        return sampleTexture(${r}, uv);
      }
    `:`
      float ${n}(int index) {
        vec2 uv = vec2((float(index + ${o}) + 0.5) / ${i}.0, 0.5);
        return sampleTexture(${r}, uv);
      }
    `:t?`
    float ${n}(int index) {
      vec2 uv = uvFromFlat(${r}TexShape[0], ${r}TexShape[1], index + ${o});
      return sampleTexture(${r}, uv);
    }
  `:`
    float ${n}(int index) {
      vec2 uv = uvFromFlat(${s}, ${i}, index + ${o});
      return sampleTexture(${r}, uv);
    }
  `}(t,r);case 2:return function(t,r){let n=t.shapeInfo.logicalShape,a=t.name,s="get"+a.charAt(0).toUpperCase()+a.slice(1),i=t.shapeInfo.texShape;if(null!=i&&nR.util.arraysEqual(n,i)){if(r)return`
      float ${s}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${a}TexShape[1], ${a}TexShape[0]);
        return sampleTexture(${a}, uv);
      }
    `;let e=i[0],t=i[1];return`
    float ${s}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${t}.0, ${e}.0);
      return sampleTexture(${a}, uv);
    }
  `}let{newShape:o,keptDims:l}=nR.util.squeezeShape(n);if(o.length<n.length){let n=xN(t,o);return`
      ${e(n,r)}
      float ${s}(int row, int col) {
        return ${s}(${x$(["row","col"],l)});
      }
    `}if(t.shapeInfo.isUniform)return`
      float ${s}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${n[1]}, 1)));
        ${xk(t)}
      }
    `;let u=i[0],h=i[1],d=xI(a);return 1===h?r?`
      float ${s}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${a}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${a}TexShape[0]));
        return sampleTexture(${a}, uv);
      }
    `:`
    float ${s}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${n[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${u}.0);
      return sampleTexture(${a}, uv);
    }
  `:1===u?r?`
      float ${s}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${a}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${a}TexShape[1]), 0.5);
        return sampleTexture(${a}, uv);
      }
    `:`
    float ${s}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${n[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${h}.0, 0.5);
      return sampleTexture(${a}, uv);
    }
  `:r?`
      float ${s}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${a}Shape[1] + col + ${d};
        vec2 uv = uvFromFlat(${a}TexShape[0], ${a}TexShape[1], index);
        return sampleTexture(${a}, uv);
      }
    `:`
  float ${s}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${n[1]} + col + ${d};
    vec2 uv = uvFromFlat(${u}, ${h}, index);
    return sampleTexture(${a}, uv);
  }
`}(t,r);case 3:return function(t,r){let n=t.shapeInfo.logicalShape,a=t.name,s="get"+a.charAt(0).toUpperCase()+a.slice(1),i=n[1]*n[2],o=n[2],{newShape:l,keptDims:u}=nR.util.squeezeShape(n);if(l.length<n.length){let n=xN(t,l);return`
        ${e(n,r)}
        float ${s}(int row, int col, int depth) {
          return ${s}(${x$(["row","col","depth"],u)});
        }
      `}if(t.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${i}, ${o}, 1)));
        ${xk(t)}
      }
    `;let h=t.shapeInfo.texShape,d=h[0],p=h[1],c=t.shapeInfo.flatOffset;if(p===i&&null==c)return r?`
      float ${s}(int row, int col, int depth) {
        int stride1 = ${a}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${a}TexShape[1], ${a}TexShape[0]);
        return sampleTexture(${a}, uv);
      }
    `:`
        float ${s}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${o}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${p}.0, ${d}.0);
          return sampleTexture(${a}, uv);
        }
      `;if(p===o&&null==c)return r?`
      float ${s}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${a}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${a}TexShape[1], ${a}TexShape[0]);
        return sampleTexture(${a}, uv);
      }
    `:`
    float ${s}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${n[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${p}.0, ${d}.0);
      return sampleTexture(${a}, uv);
    }
  `;let f=xI(a);return r?`
    float ${s}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${a}Shape[1] * ${a}Shape[2];
      int stride1 = ${a}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${f};
      vec2 uv = uvFromFlat(${a}TexShape[0], ${a}TexShape[1], index);
      return sampleTexture(${a}, uv);
    }
    `:`
      float ${s}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${i} + col * ${o} + depth + ${f};
        vec2 uv = uvFromFlat(${d}, ${p}, index);
        return sampleTexture(${a}, uv);
      }
  `}(t,r);case 4:return function(t,r){let n=t.shapeInfo.logicalShape,a=t.name,s="get"+a.charAt(0).toUpperCase()+a.slice(1),i=n[3],o=n[2]*i,l=n[1]*o,{newShape:u,keptDims:h}=nR.util.squeezeShape(n);if(u.length<n.length){let n=xN(t,u);return`
      ${e(n,r)}
      float ${s}(int row, int col, int depth, int depth2) {
        return ${s}(${x$(["row","col","depth","depth2"],h)});
      }
    `}if(t.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${l}, ${o}, ${i}, 1)));
        ${xk(t)}
      }
    `;let d=t.shapeInfo.flatOffset,p=t.shapeInfo.texShape,c=p[0],f=p[1],m=`int stride2 = ${a}Shape[3];`,g=`int stride1 = ${a}Shape[2] * stride2;`,x=`int stride0 = ${a}Shape[1] * stride1;`;if(f===l&&null==d)return r?`
      float ${s}(int row, int col, int depth, int depth2) {
        ${m}
        ${g}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${a}TexShape[1], ${a}TexShape[0]);
        return sampleTexture(${a}, uv);
      }
    `:`
      float ${s}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${o}, ${i}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${c}.0);
        return sampleTexture(${a}, uv);
      }
    `;if(f===i&&null==d)return r?`
      float ${s}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${a}Shape[1] * ${a}Shape[2], ${a}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${a}TexShape[1], ${a}TexShape[0]);
        return sampleTexture(${a}, uv);
      }
    `:`
      float ${s}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${n[1]*n[2]}, ${n[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${c}.0);
        return sampleTexture(${a}, uv);
      }
    `;let y=xI(a);return r?`
    float ${s}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${m}
      ${g}
      ${x}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${a}TexShape[0], ${a}TexShape[1], index + ${y});
      return sampleTexture(${a}, uv);
    }
  `:`
    float ${s}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${l} + col * ${o} +
          depth * ${i} + depth2;
      vec2 uv = uvFromFlat(${c}, ${f}, index + ${y});
      return sampleTexture(${a}, uv);
    }
  `}(t,r);case 5:return function(t){let r=t.shapeInfo.logicalShape,n=t.name,a="get"+n.charAt(0).toUpperCase()+n.slice(1),s=r[4],i=r[3]*s,o=r[2]*i,l=r[1]*o,{newShape:u,keptDims:h}=nR.util.squeezeShape(r);if(u.length<r.length){let r=xN(t,u);return`
      ${e(r)}
      float ${a}(int row, int col, int depth, int depth2, int depth3) {
        return ${a}(${x$(["row","col","depth","depth2","depth3"],h)});
      }
    `}if(t.shapeInfo.isUniform)return`
      float ${a}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${l}, ${o}, ${i}, ${s})) +
          depth3;
        ${xk(t)}
      }
    `;let d=t.shapeInfo.flatOffset,p=t.shapeInfo.texShape,c=p[0],f=p[1];if(f===l&&null==d)return`
      float ${a}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${o}, ${i}, ${s}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${c}.0);
        return sampleTexture(${n}, uv);
      }
    `;if(f===s&&null==d)return`
      float ${a}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${r[1]*r[2]*r[3]},
               ${r[2]*r[3]}, ${r[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${c}.0);
        return sampleTexture(${n}, uv);
      }
    `;let m=xI(n);return`
    float ${a}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${l} + col * ${o} + depth * ${i} +
          depth2 * ${s} + depth3 + ${m};
      vec2 uv = uvFromFlat(${c}, ${f}, index);
      return sampleTexture(${n}, uv);
    }
  `}(t);case 6:return function(t){let r=t.shapeInfo.logicalShape,n=t.name,a="get"+n.charAt(0).toUpperCase()+n.slice(1),{newShape:s,keptDims:i}=nR.util.squeezeShape(r);if(s.length<r.length){let r=xN(t,s);return`
      ${e(r)}
      float ${a}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${a}(${x$(["row","col","depth","depth2","depth3","depth4"],i)});
      }
    `}let o=r[5],l=r[4]*o,u=r[3]*l,h=r[2]*u,d=r[1]*h;if(t.shapeInfo.isUniform)return`
      float ${a}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${d}, ${h}, ${u}, ${l})) +
          dot(
            vec2(depth3, depth4),
            vec2(${o}, 1)));
        ${xk(t)}
      }
    `;let p=t.shapeInfo.flatOffset,c=t.shapeInfo.texShape,f=c[0],m=c[1];if(m===d&&null==p)return`
      float ${a}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${h}, ${u}, ${l}, ${o})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${m}.0, ${f}.0);
        return sampleTexture(${n}, uv);
      }
    `;if(m===o&&null==p)return`
      float ${a}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${r[1]*r[2]*r[3]*r[4]},
               ${r[2]*r[3]*r[4]},
               ${r[3]*r[4]},
               ${r[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${m}.0, ${f}.0);
        return sampleTexture(${n}, uv);
      }
    `;let g=xI(n);return`
    float ${a}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${d} + col * ${h} + depth * ${u} +
          depth2 * ${l} + depth3 * ${o} + depth4 + ${g};
      vec2 uv = uvFromFlat(${f}, ${m}, index);
      return sampleTexture(${n}, uv);
    }
  `}(t);default:throw Error(`${n.length}-D input sampling is not yet supported`)}}(e,n);let s=e.shapeInfo.logicalShape,i=t.logicalShape;return s.length<=i.length&&(r?a+=function(e,t){let r,n=e.name,a=n.charAt(0).toUpperCase()+n.slice(1),s=e.shapeInfo.logicalShape.length,i=t.logicalShape.length,o=xx(e.shapeInfo.logicalShape,t.logicalShape),l=xS(i),u=i-s,h=["x","y","z","w","u","v"];r=0===s?"":i<2&&o.length>=1?"coords = 0;":o.map(e=>`coords.${h[e+u]} = 0;`).join("\n");let d="";d=i<2&&s>0?"coords":e.shapeInfo.logicalShape.map((e,t)=>`coords.${h[t+u]}`).join(", ");let p="return outputValue;",c=1===nR.util.sizeFromShape(e.shapeInfo.logicalShape),f=1===nR.util.sizeFromShape(t.logicalShape);if(1!==s||c||f){if(c&&!f)p=1===i?`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:`
        return vec4(outputValue.x);
      `;else if(o.length){let e=s-2,t=s-1;o.indexOf(e)>-1&&o.indexOf(t)>-1?p="return vec4(outputValue.x);":o.indexOf(e)>-1?p="return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);":o.indexOf(t)>-1&&(p="return vec4(outputValue.xx, outputValue.zz);")}}else p=`
      return vec4(outputValue.xy, outputValue.xy);
    `;return`
    vec4 ${"get"+a+"AtOutCoords"}() {
      ${l} coords = getOutputCoords();
      ${r}
      vec4 outputValue = get${a}(${d});
      ${p}
    }
  `}(e,t):a+=function(e,t){let r,n=e.name,a=n.charAt(0).toUpperCase()+n.slice(1),s="get"+a+"AtOutCoords",i=t.texShape,o=e.shapeInfo.texShape,l=e.shapeInfo.logicalShape.length,u=t.logicalShape.length;if(!e.shapeInfo.isUniform&&l===u&&null==e.shapeInfo.flatOffset&&nR.util.arraysEqual(o,i))return`
      float ${s}() {
        return sampleTexture(${n}, resultUV);
      }
    `;let h=xS(u),d=xx(e.shapeInfo.logicalShape,t.logicalShape),p=u-l,c=["x","y","z","w","u","v"];r=0===l?"":u<2&&d.length>=1?"coords = 0;":d.map(e=>`coords.${c[e+p]} = 0;`).join("\n");let f="";return f=u<2&&l>0?"coords":e.shapeInfo.logicalShape.map((e,t)=>`coords.${c[t+p]}`).join(", "),`
    float ${s}() {
      ${h} coords = getOutputCoords();
      ${r}
      return get${a}(${f});
    }
  `}(e,t)),a})(e,t,r.packedInputs,r.enableShapeUniforms)).join("\n"),p=t.texShape,c=xd(),f=(n=c,`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${n.texture2D}(textureSampler, uv).r;
    }
  `),m=(a=c,`${a.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${a.varyingFs} vec2 resultUV;
    ${a.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${a.defineSpecialNaN}
    ${a.defineSpecialInf}
    ${a.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${xy}
    ${xb}
    ${xv}
  `);return t.isPacked?(o=function(e,t,r){switch(e.length){case 0:return xC();case 1:var n,a;let s;return n=t,a=r,1===(s=[Math.ceil(n[0]/2),Math.ceil(n[1]/2)])[0]?a?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${s[1]}.0);
      }
    `:1===s[1]?a?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${s[0]}.0);
      }
    `:a?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      return 2 * (resTexRC.x * ${s[1]} + resTexRC.y);
    }
  `;case 2:var i=e,o=t,l=r;let u=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)];if(nR.util.arraysEqual(i,o))return l?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${u[0]}, ${u[1]}));
      }
    `;let h=Math.ceil(i[1]/2);return l?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${u[0]}, ${u[1]}));

      int index = resTexRC.x * ${u[1]} + resTexRC.y;
      int r = 2 * (index / ${h});
      int c = imod(index, ${h}) * 2;

      return ivec2(r, c);
    }
  `;case 3:var d=e,p=t,c=r;if(c)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;let f=[Math.ceil(p[0]/2),Math.ceil(p[1]/2)],m=Math.ceil(d[2]/2),g=m*Math.ceil(d[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${f[0]}, ${f[1]}));
      int index = resTexRC.x * ${f[1]} + resTexRC.y;

      int b = index / ${g};
      index -= b * ${g};

      int r = 2 * (index / ${m});
      int c = imod(index, ${m}) * 2;

      return ivec3(b, r, c);
    }
  `;default:return function(e,t,r){if(r)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;let n=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],a=Math.ceil(e[e.length-1]/2),s=a*Math.ceil(e[e.length-2]/2),i=s,o="",l="b, r, c";for(let t=2;t<e.length-1;t++)i*=e[e.length-t-1],o=`
      int b${t} = index / ${i};
      index -= b${t} * ${i};
    `+o,l=`b${t}, `+l;return`
    ivec${e.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${n[0]}, ${n[1]}));
      int index = resTexRC.x * ${n[1]} + resTexRC.y;

      ${o}

      int b = index / ${s};
      index -= b * ${s};

      int r = 2 * (index / ${a});
      int c = imod(index, ${a}) * 2;

      return ivec${e.length}(${l});
    }
  `}(e,t,r)}}(t.logicalShape,p,r.enableShapeUniforms),s=c,l=`
    void setOutput(vec4 val) {
      ${s.output} = val;
    }
  `):(o=function(e,t,r){switch(e.length){case 0:return xC();case 1:return n=t,a=r,1===n[0]?a?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${n[1]}.0);
      }
    `:1===n[1]?a?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${n[0]}.0);
      }
    `:a?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${n[0]}, ${n[1]}));
      return resTexRC.x * ${n[1]} + resTexRC.y;
    }
  `;case 2:return s=e,i=t,o=r,nR.util.arraysEqual(s,i)?o?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${i[0]}, ${i[1]}));
      }
    `:1===s[1]?o?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${i[0]}, ${i[1]}));
        int index = resTexRC.x * ${i[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:1===s[0]?o?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${i[0]}, ${i[1]}));
        int index = resTexRC.x * ${i[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:o?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${i[0]}, ${i[1]}));
      int index = resTexRC.x * ${i[1]} + resTexRC.y;
      int r = index / ${s[1]};
      int c = index - r * ${s[1]};
      return ivec2(r, c);
    }
  `;case 3:var n,a,s,i,o,l,u,h,d,p=e,c=t,f=r;if(f){let e=xc(["r","c","d"],p);return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${e}
    return ivec3(r, c, d);
  }
`}let m=xp(["r","c","d"],p);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${c[0]}, ${c[1]}));
      int index = resTexRC.x * ${c[1]} + resTexRC.y;
      ${m}
      return ivec3(r, c, d);
    }
  `;case 4:var g=e,x=t,y=r;if(y){let e=xc(["r","c","d","d2"],g);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${e}
      return ivec4(r, c, d, d2);
    }
  `}let b=xp(["r","c","d","d2"],g);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${x[0]}, ${x[1]}));
      int index = resTexRC.x * ${x[1]} + resTexRC.y;
      ${b}
      return ivec4(r, c, d, d2);
    }
  `;case 5:let v;return l=e,u=t,v=xp(["r","c","d","d2","d3"],l),`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${u[0]},
                             ${u[1]}));

      int index = resTexRC.x * ${u[1]} + resTexRC.y;

      ${v}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `;case 6:let w;return h=e,d=t,w=xp(["r","c","d","d2","d3","d4"],h),`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${d[0]}, ${d[1]}));
      int index = resTexRC.x * ${d[1]} + resTexRC.y;

      ${w}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `;default:throw Error(`${e.length}-D output sampling is not yet supported`)}}(t.logicalShape,p,r.enableShapeUniforms),i=c,l=`
    void setOutput(float val) {
      ${i.output} = vec4(val, 0, 0, 0);
    }
  `),r.packedInputs&&(m+=xw),[m,f,l,h,o,d,r.userCode].join("\n")}(r,a={logicalShape:f.shape,texShape:f.texData.texShape,isUniform:!1,isPacked:f.texData.isPacked,flatOffset:null},e),i=gz(t.gl,s),o=t.createProgram(i),(0,nN.env)().get("ENGINE_COMPILE_ONLY")?{program:e,fragmentShader:i,source:s,webGLProgram:o,inShapeInfos:n,outShapeInfo:a,variablesLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:(t.buildVao(o),Object.assign({program:e,fragmentShader:i,source:s,webGLProgram:o,inShapeInfos:n,outShapeInfo:a},xR(t,e,o)))}),x=null!=this.activeTimers;x&&(i=this.startTimer()),(0,nN.env)().get("ENGINE_COMPILE_ONLY")||function(e,t,r,n,a){t.program.enableShapeUniforms||(xA(t.inShapeInfos,r),xA([t.outShapeInfo],[n]));let s=n.texData.texture,i=n.texData.texShape;n.texData.isPacked?e.setOutputPackedMatrixTexture(s.texture,i[0],i[1]):e.setOutputMatrixTexture(s.texture,i[0],i[1]),e.setProgram(t.webGLProgram),e.bindVertexArray(t.webGLProgram.vao),1===(0,nN.env)().getNumber("WEBGL_VERSION")&&null!==t.infLoc&&e.gl.uniform1f(t.infLoc,1/0),null!==t.nanLoc&&e.gl.uniform1f(t.nanLoc,NaN);for(let n=0;n<r.length;++n){let a=r[n],{uniform:s,offset:i,shape:o,texShape:l}=t.variablesLocations[n];if(o){let{uniformShape:r}=xT(t.program.packedInputs,a.shape,a.texData.texShape);switch(r.length){case 1:e.gl.uniform1iv(o,new Int32Array(r));break;case 2:e.gl.uniform2iv(o,new Int32Array(r));break;case 3:e.gl.uniform3iv(o,new Int32Array(r));break;case 4:e.gl.uniform4iv(o,new Int32Array(r))}}if(l&&e.gl.uniform2i(l,a.texData.texShape[0],a.texData.texShape[1]),null!=s){if(a.isUniform){if(2>nR.util.sizeFromShape(a.shape))e.gl.uniform1f(s,a.uniformValues[0]);else{let t=a.uniformValues;t instanceof Float32Array||(t=new Float32Array(t)),e.gl.uniform1fv(s,t)}continue}null!=a.texData.slice&&null!=i&&e.gl.uniform1i(i,a.texData.slice.flatOffset),e.setInputMatrixTexture(a.texData.texture.texture,s,n)}}let o=t.outShapeLocation;if(o)switch(n.shape.length){case 1:e.gl.uniform1iv(o,new Int32Array(n.shape));break;case 2:e.gl.uniform2iv(o,new Int32Array(n.shape));break;case 3:e.gl.uniform3iv(o,new Int32Array(n.shape));break;case 4:e.gl.uniform4iv(o,new Int32Array(n.shape))}if(t.outShapeStridesLocation){let r=nR.util.computeStrides(n.shape);switch(n.shape.length){case 2:e.gl.uniform1iv(t.outShapeStridesLocation,new Int32Array(r));break;case 3:e.gl.uniform2iv(t.outShapeStridesLocation,new Int32Array(r));break;case 4:e.gl.uniform3iv(t.outShapeStridesLocation,new Int32Array(r))}}if(t.outTexShapeLocation&&e.gl.uniform2i(t.outTexShapeLocation,n.texData.texShape[0],n.texData.texShape[1]),t.program.customUniforms&&a)for(let r=0;r<t.program.customUniforms.length;++r){let n=t.program.customUniforms[r],s=t.customUniformLocations[r],i=a[r];if("float"===n.type)e.gl.uniform1fv(s,i);else if("vec2"===n.type)e.gl.uniform2fv(s,i);else if("vec3"===n.type)e.gl.uniform3fv(s,i);else if("vec4"===n.type)e.gl.uniform4fv(s,i);else if("int"===n.type)e.gl.uniform1iv(s,i);else if("ivec2"===n.type)e.gl.uniform2iv(s,i);else if("ivec3"===n.type)e.gl.uniform3iv(s,i);else if("ivec4"===n.type)e.gl.uniform4iv(s,i);else throw Error(`uniform type ${n.type} is not supported yet.`)}e.executeProgram()}(this.gpgpu,g,p,f,n),d.forEach(e=>this.disposeIntermediateTensorInfo(e)),x&&(i=this.endTimer(i),this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime(i)}));let y=(0,nN.env)().getNumber("WEBGL_FLUSH_THRESHOLD");if(y>0){let e=nR.util.now();e-this.lastGlFlushTime>y&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=e)}if(!(0,nN.env)().getBool("WEBGL_LAZILY_UNPACK")&&h.isPacked&&!1===a){let e=this.unpackTensor(u);return this.disposeIntermediateTensorInfo(u),e}return u}compileAndRun(e,t,r,n,a=!1){return r=r||t[0].dtype,this.runWebGLProgram(e,t,r,n,a)}getAndSaveBinary(e,t){return e in this.binaryCache||(this.binaryCache[e]=t()),this.binaryCache[e]}getTextureManager(){return this.textureManager}dispose(){this.disposed||((0,nN.env)().getBool("IS_TEST")||Object.keys(this.binaryCache).forEach(e=>{this.gpgpu.deleteProgram(this.binaryCache[e].webGLProgram),delete this.binaryCache[e]}),this.textureManager.dispose(),null!=this.canvas&&"u">typeof HTMLCanvasElement&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),this.disposed=!0)}floatPrecision(){return null==this.floatPrecisionValue&&(this.floatPrecisionValue=(0,n$.tidy)(()=>{if(!(0,nN.env)().get("WEBGL_RENDER_FLOAT32_ENABLED")){let e=(0,nN.env)().getBool("DEBUG");(0,nN.env)().set("DEBUG",!1);let t=this.abs((0,k.scalar)(1e-8)).dataSync()[0];if((0,nN.env)().set("DEBUG",e),t>0)return 32}return 16})),this.floatPrecisionValue}epsilon(){return 32===this.floatPrecision()?1e-7:1e-4}uploadToGPU(e){let t,r=this.texData.get(e),{shape:n,dtype:a,values:s,texture:i,usage:o,isPacked:l}=r;if(null!=i)return;let u=null!=this.activeTimers;u&&(t=nR.util.now());let h=r.texShape;if(null==h&&(r.texShape=h=g8(n,l)),null!=s){let e,i=g6(n),o=h[1],d=h[0],p=s instanceof Uint8Array||s instanceof Uint8ClampedArray;(l||!p)&&([o,d]=gR(h[0],h[1])),e=l?new xM(i,p):new x_(i,p);let c=p?[d,o]:h,m=this.makeTensorInfo(c,a),g=this.texData.get(m.dataId);p?g.usage=f.PIXELS:g.usage=f.UPLOAD,g.texShape=c,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(m.dataId),o,d,s);let x=[[d,o]],y=this.runWebGLProgram(e,[m],a,x,!0),b=this.texData.get(y.dataId);r.texShape=b.texShape,r.isPacked=b.isPacked,r.usage=b.usage,(0,nN.env)().get("ENGINE_COMPILE_ONLY")?this.disposeData(y.dataId):(r.texture=b.texture,r.values=null,this.texData.delete(y.dataId)),this.disposeIntermediateTensorInfo(m),u&&(this.uploadWaitMs+=nR.util.now()-t)}else r.texture=this.acquireTexture(h,o,a,l)}convertAndCacheOnCPU(e,t){let r=this.texData.get(e),{dtype:n}=r;return null!=t&&(r.values=function(e,t){if("float32"===t||"complex64"===t)return e;if("int32"===t||"bool"===t){let r="int32"===t?new Int32Array(e.length):new Uint8Array(e.length);for(let t=0;t<r.length;++t)r[t]=Math.round(e[t]);return r}throw Error(`Unknown dtype ${t}`)}(t,n)),r.values}acquireTexture(e,t,r,n){if(this.numBytesInGPU+=this.computeBytes(e,r),!this.warnedAboutMemory&&this.numBytesInGPU>1024*this.numMBBeforeWarning*1024){let e=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${e} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(e,t,n)}computeBytes(e,t){return e[0]*e[1]*nR.util.bytesPerElement(t)}checkCompileCompletion(){for(let[,e]of Object.entries(this.binaryCache))this.checkCompletion_(e)}async checkCompileCompletionAsync(){let e=[];if(this.gpgpu.parallelCompilationExtension){for(let[,t]of Object.entries(this.binaryCache))e.push(this.checkCompletionAsync_(t));return Promise.all(e)}for(let[,t]of Object.entries(this.binaryCache)){let r=new Promise(e=>{try{this.checkCompletion_(t),e(!0)}catch(e){throw e}});e.push(r)}return Promise.all(e)}async checkCompletionAsync_(e){return this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(e):(await (0,sz.nextFrame)(),this.checkCompletionAsync_(e))}checkCompletion_(e){if(!1===this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.gl.LINK_STATUS)){if(console.log(this.gpgpu.gl.getProgramInfoLog(e.webGLProgram)),!1===this.gpgpu.gl.getShaderParameter(e.fragmentShader,this.gpgpu.gl.COMPILE_STATUS))throw gM(e.source,this.gpgpu.gl.getShaderInfoLog(e.fragmentShader)),Error("Failed to compile fragment shader.");throw Error("Failed to link vertex and fragment shaders.")}return!0}getUniformLocations(){for(let e of Object.values(this.binaryCache)){this.gpgpu.buildVao(e.webGLProgram);let{variablesLocations:t,customUniformLocations:r,infLoc:n,nanLoc:a,outShapeLocation:s,outShapeStridesLocation:i,outTexShapeLocation:o}=xR(this.gpgpu,e.program,e.webGLProgram);e.variablesLocations=t,e.customUniformLocations=r,e.infLoc=n,e.nanLoc=a,e.outShapeLocation=s,e.outShapeStridesLocation=i,e.outTexShapeLocation=o}}createTensorFromGPUData(e,t,r){e.channels=e.channels||"RGBA";let{texture:n,height:a,width:s,channels:i}=e,o=(0,n$.engine)().backend;if(!o.gpgpu.gl.isTexture(n))throw Error("The texture is invalid. Also, please make sure the texture and the TFJS WebGL backend are using the same canvas. If you want to use your own custom canvas, you have to create and use the custom TFJS WebGL backend created from the canvas through 'new tf.MathBackendWebGL(customCanvas)'.");let l=o.writeTexture(n,t,r,a,s,i);return(0,n$.engine)().makeTensorFromDataId(l,t,r,o)}}bs.nextDataId=0;var bi=e.i(86136),bo=e.i(71065);function bl(){(0,nN.env)().set("WEBGL_FORCE_F16_TEXTURES",!0)}e.s(["forceHalfFloat",0,bl],69754),gI.isBrowser()&&(0,n$.registerBackend)("webgl",()=>new bs,2);let bu={forceHalfFloat:bl};e.s(["webgl",0,bu],51868);var cy=A,tN=tN,tN=tN;let bh=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;class bd{constructor(e,t,r){this.variableNames=["A","B"],this.outputShape=tN.assertAndGetBroadcastShape(t,r),this.enableShapeUniforms=xE(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${e}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}}var tN=tN;let bp=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`;class bc{constructor(e,t,r,n=!1){this.variableNames=["A","B"],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=tN.assertAndGetBroadcastShape(t,r);const a=this.outputShape.length;this.enableShapeUniforms=xE(a);let s="";if(n)if(0===a||1===nR.util.sizeFromShape(this.outputShape))s=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else{const e=xS(a);if(s=`
          ${e} coords = getOutputCoords();
        `,1===a)this.enableShapeUniforms?s+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:s+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{const e=yX("coords",a);this.enableShapeUniforms?s+=`
            bool nextRowOutOfBounds =
              (${e[a-2]} + 1) >= outShape[${a} - 2];
            bool nextColOutOfBounds =
              (${e[a-1]} + 1) >= outShape[${a} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:s+=`
            bool nextRowOutOfBounds =
              (${e[a-2]} + 1) >= ${this.outputShape[a-2]};
            bool nextColOutOfBounds =
              (${e[a-1]} + 1) >= ${this.outputShape[a-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${e}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${s}

        setOutput(result);
      }
    `}}function bf(e){let{inputs:t,backend:r}=e,{x:n}=t;return r.incRef(n.dataId),{dataId:n.dataId,shape:n.shape,dtype:n.dtype}}let bm={kernelName:x.Identity,backendName:"webgl",kernelFunc:bf};function bg(e){let{inputs:t,backend:r}=e,{real:n,imag:a}=t,s=r.makeTensorInfo(n.shape,"complex64");return r.texData.get(s.dataId).complexTensorInfos={real:bf({inputs:{x:n},backend:r}),imag:bf({inputs:{x:a},backend:r})},s}let bx={kernelName:x.Complex,backendName:"webgl",kernelFunc:bg},by="return (a < 0.) ? b * a : a;",bb=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`,bv={kernelName:x.LeakyRelu,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{alpha:s}=n,i=r.makeTensorInfo([],"float32",nR.util.createScalarValue(s,"float32")),o=(0,nN.env)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new bc(bb,a.shape,i.shape):new bd(by,a.shape,i.shape),l=r.runWebGLProgram(o,[a,i],"float32");return r.disposeIntermediateTensorInfo(i),l}},bw="return (a < 0.) ? b * a : a;",bC=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`,bI={kernelName:x.Prelu,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{x:n,alpha:a}=t,s=(0,nN.env)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new bc(bC,n.shape,a.shape):new bd(bw,n.shape,a.shape);return r.runWebGLProgram(s,[n,a],"float32")}},bk="if (isnan(x)) return x;";function bS({opSnippet:e,packedOpSnippet:t,cpuKernelImpl:r,dtype:n}){return({inputs:a,backend:s})=>{let i,{x:o}=a,l=n||o.dtype;if(s.shouldExecuteOnCPU([o])&&null!=r){let e=r(s.texData.get(o.dataId).values,l);return s.makeTensorInfo(o.shape,l,e)}return i=(0,nN.env)().getBool("WEBGL_PACK_UNARY_OPERATIONS")&&null!=t?new be(o.shape,t):new y1(o.shape,e),s.runWebGLProgram(i,[o],l)}}function bT({opSnippet:e,packedOpSnippet:t,checkOutOfBounds:r=!1,supportsComplex:n=!1,cpuKernelImpl:a,dtype:s}){return({inputs:i,backend:o})=>{let l,{a:u,b:h}=i;if(n&&"complex64"===u.dtype){let t=o.texData.get(u.dataId),r=o.texData.get(h.dataId),[n,a]=[[t.complexTensorInfos.real,r.complexTensorInfos.real],[t.complexTensorInfos.imag,r.complexTensorInfos.imag]].map(t=>{let[r,n]=t,a={dataId:r.dataId,dtype:r.dtype,shape:u.shape},s={dataId:n.dataId,dtype:n.dtype,shape:h.shape},i=new bd(e,u.shape,h.shape);return o.runWebGLProgram(i,[a,s],(0,py.upcastType)(r.dtype,n.dtype))}),s=bg({inputs:{real:n,imag:a},backend:o});return o.disposeIntermediateTensorInfo(n),o.disposeIntermediateTensorInfo(a),s}let d=s||(0,py.upcastType)(u.dtype,h.dtype);if(("string"===u.dtype||"string"===h.dtype||o.shouldExecuteOnCPU([u,h]))&&null!=a){let e=o.texData.get(u.dataId).values,t=o.texData.get(h.dataId).values,r="string"===u.dtype?tN.fromUint8ToStringArray(e):e,n="string"===u.dtype?tN.fromUint8ToStringArray(t):t,[s,i]=a(u.shape,h.shape,r,n,d),l=o.makeTensorInfo(i,d);return o.texData.get(l.dataId).values=s,l}return l=(0,nN.env)().getBool("WEBGL_PACK_BINARY_OPERATIONS")&&null!=t?new bc(t,u.shape,h.shape,r):new bd(e,u.shape,h.shape),o.runWebGLProgram(l,[u,h],d)}}function bN(e,t=!1){if("linear"===e)return"return x;";if("relu"===e)return t?y7:y4;if("elu"===e)return t?y8:"return (x >= 0.0) ? x : (exp(x) - 1.0);";if("relu6"===e)return t?y9:y5;if("prelu"===e)return t?bC:bw;else if("leakyrelu"===e)return t?bb:by;else if("sigmoid"===e)return"return 1.0 / (1.0 + exp(-1.0 * x));";throw Error(`Activation ${e} has not been implemented for the WebGL backend.`)}class b${constructor(e,t,r,n=!1,a=!1,s=!1,i=null,o=!1,l=!1){this.variableNames=["matrixA","matrixB"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=r,this.enableShapeUniforms=xE(this.outputShape.length);const u=Math.ceil((n?e[1]:e[2])/2),h=n?["a.xxyy","a.zzww"]:["a.xxzz","a.yyww"],d=a?["b.xzxz","b.ywyw"]:["b.xyxy","b.zwzw"];let p="",c="";i&&(p=o?`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${i}
        }`:l?`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${i}
        }`:`vec4 activation(vec4 x) {
          ${i}
        }`,c="result = activation(result);"),s&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),l&&this.variableNames.push("leakyreluAlpha");let f="rc.x",m="rc.x";e[0]<t[0]?f=`imod(rc.x, ${e[0]})`:t[0]<e[0]&&(m=`imod(rc.x, ${t[0]})`),this.userCode=`
      ${p}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${u}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${f};
        int batchB = ${m};
        for (int i = 0; i < ${u}; i++) {
          vec4 a = getMatrixA(batchA, ${n?"i * 2, rc.y":"rc.y, i * 2"});
          vec4 b = getMatrixB(batchB, ${a?"rc.z, i * 2":"i * 2, rc.z"});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${h[0]} * ${d[0]});
          result += (${h[1]} * ${d[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${s?"result += getBiasAtOutCoords();":""}

        ${c}

        setOutput(result);
      }
    `}}var tN=tN,tN=tN;class bR{constructor(e,t,r){this.variableNames=["AReal","AImag","BReal","BImag"],this.outputShape=tN.assertAndGetBroadcastShape(t,r),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${e}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}}let bA="return a * b;";function bE(e){let t,{inputs:r,backend:n}=e,{a,b:s}=r,i=tN.upcastType(a.dtype,s.dtype);if("complex64"===a.dtype){let e=n.texData.get(a.dataId),t=n.texData.get(s.dataId),r=new bR("return areal * breal - aimag * bimag;",a.shape,s.shape),i=new bR("return areal * bimag + aimag * breal;",a.shape,s.shape),o=[{dataId:e.complexTensorInfos.real.dataId,dtype:e.complexTensorInfos.real.dtype,shape:a.shape},{dataId:e.complexTensorInfos.imag.dataId,dtype:e.complexTensorInfos.imag.dtype,shape:a.shape},{dataId:t.complexTensorInfos.real.dataId,dtype:t.complexTensorInfos.real.dtype,shape:s.shape},{dataId:t.complexTensorInfos.imag.dataId,dtype:t.complexTensorInfos.imag.dtype,shape:s.shape}],l=n.runWebGLProgram(r,o,"float32"),u=n.runWebGLProgram(i,o,"float32"),h=bg({inputs:{real:l,imag:u},backend:n});return n.disposeIntermediateTensorInfo(l),n.disposeIntermediateTensorInfo(u),h}if(n.shouldExecuteOnCPU([a,s])){let e=n.texData.get(a.dataId),t=n.texData.get(s.dataId),[r,o]=yv(a.shape,s.shape,e.values,t.values,i),l=n.makeTensorInfo(o,i);return n.texData.get(l.dataId).values=r,l}return t=(0,nN.env)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new bc(bA,a.shape,s.shape):new bd(bA,a.shape,s.shape),n.runWebGLProgram(t,[a,s],i)}let bF={kernelName:x.Multiply,backendName:"webgl",kernelFunc:bE};function bD(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{shape:s}=n,i=nR.util.sizeFromShape(a.shape),o=nR.util.inferFromImplicitShape(s,i),l=nR.util.sizeFromShape(o);nR.util.assert(i===l,()=>`The new shape (${o}) has ${l} elements and the old shape (${a.shape}) has ${i} elements. The new shape and old shape must have the same number of elements.`);let u=r.texData.get(a.dataId);if(u.isPacked&&!g7(a.shape,o)&&!(null!==u.texture&&g7(u.shape,o))){let e,t,n,s,i;return e=[g4(a.shape),...g5(a.shape)],t={dtype:a.dtype,shape:e,dataId:a.dataId},n=new yY([g4(o),...g5(o)],e),s=[e],{dataId:(i=r.runWebGLProgram(n,[t],a.dtype,s,!0)).dataId,shape:o,dtype:i.dtype}}return r.incRef(a.dataId),{dataId:a.dataId,shape:o,dtype:a.dtype}}let bO={kernelName:x.Reshape,backendName:"webgl",kernelFunc:bD};var tN=tN,tN=tN;class bL{constructor(e,t){this.variableNames=["x"];const{windowSize:r,batchSize:n,inSize:a,outSize:s}=e;this.outputShape=[n,s];const i=4*Math.floor(r/4),o=r%4;let l="sumValue += dot(values, ones);";if(null!=t){const e=1/t;l=`sumValue += dot(values * ${nR.util.isInt(e)?e.toPrecision(2):e}, ones);`}let u="";a%r>0&&(u=`
        if (inIdx < 0 || inIdx >= ${a}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${r};

        float sumValue = 0.0;

        for (int i = 0; i < ${i}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${l}
        }

        int inIdx = inOffset + ${i};
        if (${1===o}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${l}
        } else if (${2===o}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${l}
        } else if (${3===o}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${l}
        }
        setOutput(sumValue);
      }
    `}}class bz{constructor(e,t){this.variableNames=["x"];const{windowSize:r,batchSize:n,inSize:a,outSize:s}=e;this.outputShape=[n,s];let i="0.0",o="";"prod"===t?i="1.0":"min"===t?(i="1.0 / 1e-20",o="min"):"max"===t&&(i="-1.0 / 1e-20",o="max");let l=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;"sum"===t?l="sumValue":"prod"===t?l="prodValue":"all"===t?l="allValue":"any"===t&&(l="anyValue");const u=4*Math.floor(r/4),h=r%4;let d=`
      if (${"sum"===t}) {
        sumValue += dot(values, ones);
      } else if (${"prod"===t}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${o}(values, minMaxValue);
        if (${"min"===t} || ${"max"===t}) {
          minMaxValue = ${o}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,p="vec4";"all"===t?(i="1.0",d=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,p="bvec4"):"any"===t&&(i="0.0",d=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,p="bvec4");let c="";a%r>0&&(c=`
        if (inIdx < 0 || inIdx >= ${a}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${i};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${c}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${r};

        vec4 minMaxValue = vec4(${i});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${d}
        }

        int inIdx = inOffset + ${u};
        if (${1===h}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${2===h}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${3===h}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${d}
        }
        setOutput(${l});
      }
    `}}function b_(e,t,r,n){let a=function(e){let t=[];for(;0===t.length||1!==t[t.length-1].outSize;){let r=t.length?t[t.length-1].outSize:e[1],n=tN.computeOptimalWindowSize(r);t.push({inSize:r,windowSize:n,outSize:Math.ceil(r/n)})}return t}(e.shape),s=e;for(let i=0;i<a.length;i++){let o,l,{inSize:u,windowSize:h,outSize:d}=a[i];o="mean"===r?0===i?new bL({windowSize:h,inSize:u,batchSize:e.shape[0],outSize:d},u):new bL({windowSize:h,inSize:u,batchSize:e.shape[0],outSize:d}):new bz({windowSize:h,inSize:u,batchSize:e.shape[0],outSize:d},r),l=s,s=n.runWebGLProgram(o,[s],t),l.dataId!==e.dataId&&n.disposeIntermediateTensorInfo(l)}return s}class bM{constructor(e,t){this.variableNames=["A"];const r=Array(e.length);for(let n=0;n<r.length;n++)r[n]=e[t[n]];this.outputShape=r,this.rank=r.length;const n=xS(this.rank),a=function(e){let t=e.length;if(t>6)throw Error(`Transpose for rank ${t} is not yet supported`);let r=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u","resRC.v"],n=Array(t);for(let t=0;t<e.length;t++)n[e[t]]=r[t];return n.join()}(t);this.userCode=`
    void main() {
      ${n} resRC = getOutputCoords();
      setOutput(getA(${a}));
    }
    `}}class bP{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0;const r=Array(e.length);for(let n=0;n<r.length;n++)r[n]=e[t[n]];if(this.outputShape=r,this.rank=r.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);const n=xS(this.rank),a=yj("rc",this.rank),s=Array(this.rank);for(let e=0;e<t.length;e++)s[t[e]]=a[e];const i=`vec2(${s.slice(-2).join()})`,o=`++${a[this.rank-1]} < ${r[this.rank-1]}`,l=`getChannel(getA(${s.join()}), ${i})`;this.userCode=`
    void main() {
      ${n} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${l};
      if(${o}) {
        result[1] = ${l};
      }
      --${a[this.rank-1]};
      if(++${a[this.rank-2]} < ${r[this.rank-2]}) {
        result[2] = ${l};
        if(${o}) {
          result[3] = ${l};
        }
      }
      setOutput(result);
    }
    `}}function bB(e,t,r){let n=(0,nN.env)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new bP(e.shape,t):new bM(e.shape,t);return r.runWebGLProgram(n,[e],e.dtype)}function bW(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,keepDims:i}=n;return function(e,t,r,n){let a=e.shape.length,s=nR.util.parseAxisParam(t,e.shape),i=s,o=tN.getAxesPermutation(i,a),l=null!=o,u=e;l&&(u=bB(e,o,n),i=tN.getInnerMostAxes(i.length,a)),tN.assertAxesAreInnerMostDims("sum",i,a);let[h,d]=tN.computeOutAndReduceShapes(u.shape,i),p=h;r&&(p=tN.expandShapeToKeepDim(h,s));let c=nR.util.sizeFromShape(d),f=bD({inputs:{x:u},attrs:{shape:[nR.util.sizeFromShape(e.shape)/c,c]},backend:n}),m=b_(f,(0,py.sumOutType)(e.dtype),"sum",n),g=bD({inputs:{x:m},attrs:{shape:p},backend:n});return n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(m),l&&n.disposeIntermediateTensorInfo(u),g}(a,s,i,r)}let bG={kernelName:x.Sum,backendName:"webgl",kernelFunc:bW};function bU(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{perm:i}=a,o=Array(s.shape.length);for(let e=0;e<o.length;e++)o[e]=s.shape[i[e]];if(n.shouldExecuteOnCPU([s])){let e=yH(n.texData.get(s.dataId).values,s.shape,s.dtype,i,o);t=n.makeTensorInfo(o,s.dtype),n.texData.get(t.dataId).values=e}else t=bB(s,i,n);return t}let bV={kernelName:x.Transpose,backendName:"webgl",kernelFunc:bU};function bH({a:e,b:t,transposeA:r,transposeB:n,backend:a,bias:s=null,preluActivationWeights:i=null,leakyreluAlpha:o=0,activation:l=null}){let u,h=e.shape.length,d=t.shape.length,p=r?e.shape[h-2]:e.shape[h-1],c=n?t.shape[d-1]:t.shape[d-2],f=r?e.shape[h-1]:e.shape[h-2],m=n?t.shape[d-2]:t.shape[d-1],g=e.shape.slice(0,-2),x=t.shape.slice(0,-2),y=nR.util.sizeFromShape(g),b=nR.util.sizeFromShape(x),v=cy.assertAndGetBroadcastShape(e.shape.slice(0,-2),t.shape.slice(0,-2)).concat([f,m]);nR.util.assert(p===c,()=>`Error in matMul: inner shapes (${p}) and (${c}) of Tensors with shapes ${e.shape} and ${t.shape} and transposeA=${r} and transposeB=${n} must match.`);let w=r?[y,p,f]:[y,f,p],C=n?[b,m,c]:[b,c,m],I=bD({inputs:{x:e},backend:a,attrs:{shape:w}}),k=bD({inputs:{x:t},backend:a,attrs:{shape:C}}),S=[I,k],T=Math.max(y,b),N=r?I.shape[1]:I.shape[2],$=null!=s,R=null!=i,A="leakyrelu"===l,E=null!=l?bN(l,!0):null,F=$||R||A||null!=E;if((1===f||1===m)&&N>1e3&&!1===F){let e=I,t=k;r&&(e=bU({inputs:{x:I},backend:a,attrs:{perm:[0,2,1]}}),S.push(e)),n&&(t=bU({inputs:{x:k},backend:a,attrs:{perm:[0,2,1]}}),S.push(t));let s=1!==m,i=1===m,o=e;s&&(o=bD({inputs:{x:e},backend:a,attrs:{shape:[T,N,1]}}),S.push(o));let l=t;i&&(l=bD({inputs:{x:t},backend:a,attrs:{shape:[T,1,N]}}),S.push(l));let h=bE({inputs:{a:o,b:l},backend:a});u=bW({inputs:{x:h},backend:a,attrs:{axis:1===m?2:1,keepDims:!0}}),S.push(h)}else{let l=(0,py.upcastType)(e.dtype,t.dtype),h=new b$(w,C,[T,f,m],r,n,$,E,R,A),d=[I,k];if(null!=s&&d.push(s),R&&d.push(i),A){let e=a.makeTensorInfo([],"float32",nR.util.createScalarValue(o,"float32"));d.push(e),S.push(e)}u=a.runWebGLProgram(h,d,l)}let D=bD({inputs:{x:u},backend:a,attrs:{shape:v}});for(let e of(S.push(u),S))a.disposeIntermediateTensorInfo(e);return D}let bq={kernelName:x._FusedMatMul,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{a,b:s,bias:i,preluActivationWeights:o}=t,{transposeA:l,transposeB:u,activation:h,leakyreluAlpha:d}=n;return bH({a,b:s,transposeA:l,transposeB:u,backend:r,bias:i,preluActivationWeights:o,leakyreluAlpha:d,activation:h})}},bj="return abs(x);",bX={kernelName:x.Abs,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n}=e,{x:a}=r;if(n.shouldExecuteOnCPU([a])&&"complex64"!==a.dtype){let e=yE(n.texData.get(a.dataId).values);return n.makeTensorInfo(a.shape,a.dtype,e)}return t=(0,nN.env)().getBool("WEBGL_PACK_UNARY_OPERATIONS")?new be(a.shape,bj):new y1(a.shape,bj),n.runWebGLProgram(t,[a],a.dtype)}},bK=bS({opSnippet:y2+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`}),bY={kernelName:x.Acos,backendName:"webgl",kernelFunc:bK},bZ=bS({opSnippet:y2+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`}),bJ={kernelName:x.Acosh,backendName:"webgl",kernelFunc:bZ},bQ="return a + b;",b0=bT({opSnippet:bQ,packedOpSnippet:bQ,supportsComplex:!0,cpuKernelImpl:x7}),b1={kernelName:x.Add,backendName:"webgl",kernelFunc:b0};class b2{constructor(e,t){this.outputShape=[],this.outputShape=e,this.variableNames=t.map((e,t)=>`T${t}`);const r=[];this.variableNames.forEach(e=>{r.push(`float v${e} = get${e}AtOutCoords();`)});const n=this.variableNames.map(e=>`v${e}`).join(" + ");this.userCode=`
      void main() {
        ${r.join("\n        ")}

        float result = ${n};
        setOutput(result);
      }
    `}}class b3{constructor(e,t){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.variableNames=t.map((e,t)=>`T${t}`);const r=[];this.variableNames.forEach(e=>{r.push(`vec4 v${e} = get${e}AtOutCoords();`)});const n=this.variableNames.map(e=>`v${e}`).join(" + ");this.userCode=`
      void main() {
        ${r.join("\n        ")}

        vec4 result = ${n};
        setOutput(result);
      }
    `}}let b4={kernelName:x.AddN,backendName:"webgl",kernelFunc:function e(t){let{inputs:r,backend:n}=t;if(1===r.length)return bf({inputs:{x:r[0]},backend:n});if(r.length>(0,nN.env)().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER")){let t=Math.floor(r.length/2),a=e({inputs:r.slice(0,t),backend:n}),s=e({inputs:r.slice(t),backend:n});return e({inputs:[a,s],backend:n})}let a=r.map(e=>e.dtype).reduce((e,t)=>(0,py.upcastType)(e,t)),s=r.map(e=>e.shape),i=(0,nN.env)().getBool("WEBGL_PACK")?new b3(r[0].shape,s):new b2(r[0].shape,s);return n.runWebGLProgram(i,r,a)}};var tN=tN;let b5={kernelName:x.All,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{axis:i,keepDims:o}=a,l=s.shape.length,u=nR.util.parseAxisParam(i,s.shape),h=u,d=tN.getAxesPermutation(h,l),p=s;null!=d&&(p=bU({inputs:{x:s},backend:n,attrs:{perm:d}}),h=tN.getInnerMostAxes(h.length,l)),tN.assertAxesAreInnerMostDims("all",h,l);let[c,f]=tN.computeOutAndReduceShapes(p.shape,h),m=bD({inputs:{x:p},backend:n,attrs:{shape:[-1,nR.util.sizeFromShape(f)]}}),g=b_(m,m.dtype,"all",n);return t=o?bD({inputs:{x:g},backend:n,attrs:{shape:tN.expandShapeToKeepDim(c,u)}}):bD({inputs:{x:g},backend:n,attrs:{shape:c}}),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(g),null!=d&&n.disposeIntermediateTensorInfo(p),t}};var tN=tN;let b6={kernelName:x.Any,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{axis:i,keepDims:o}=a,l=s.shape.length,u=nR.util.parseAxisParam(i,s.shape),h=u,d=tN.getAxesPermutation(h,l),p=s;null!=d&&(p=bU({inputs:{x:s},backend:n,attrs:{perm:d}}),h=tN.getInnerMostAxes(h.length,l)),tN.assertAxesAreInnerMostDims("any",h,l);let[c,f]=tN.computeOutAndReduceShapes(p.shape,h),m=bD({inputs:{x:p},backend:n,attrs:{shape:[-1,nR.util.sizeFromShape(f)]}}),g=b_(m,m.dtype,"any",n);return t=o?bD({inputs:{x:g},backend:n,attrs:{shape:tN.expandShapeToKeepDim(c,u)}}):bD({inputs:{x:g},backend:n,attrs:{shape:c}}),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(g),null!=d&&n.disposeIntermediateTensorInfo(p),t}};var tN=tN,tN=tN;class b8{constructor(e,t,r){this.variableNames=["A"];const{windowSize:n,batchSize:a,outSize:s}=e;r||this.variableNames.push("bestIndicesA"),this.outputShape=[a,s],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${n};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${n}; i++) {
          int inIdx = ${r?"inOffset + i;":"round(getBestIndicesA(batch, inOffset + i));"};
          float candidate = getA(batch, inIdx);
          if (candidate ${"max"===t?">":"<"} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}}class b7{constructor(e,t,r,n){let a,s;this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,nR.util.assert(e.length>2,()=>`Packed arg${r.charAt(0).toUpperCase()+r.slice(1)} supports only inputs with rank above 2.`);const i=Math.ceil(e[e.length-1]/t);this.outputShape=e.slice(0,-1),i>1&&this.outputShape.push(i),n||this.variableNames.push("bestIndicesA");const o=this.outputShape,l=o.length,u=xS(l),h=yX("coords",l);if(1===i){const e=xS(s=l+1);a=`
        ${e} sourceLocR = ${e}(${h.join()}, 0);
        ++${h[l-1]};
        ${e} sourceLocG = ${e}(${h.join()}, 0);
        ++${h[l-2]};
        ${e} sourceLocA = ${e}(${h.join()}, 0);
        --${h[l-1]};
        ${e} sourceLocB = ${e}(${h.join()}, 0);
        --${h[l-2]};`}else s=l,a=`
        ${u} sourceLocR = coords;
        ++${h[l-1]};
        ${u} sourceLocG = coords;
        ++${h[l-2]};
        ${u} sourceLocA = coords;
        --${h[l-1]};
        ${u} sourceLocB = coords;
        --${h[l-2]};`;const d=["x","y","z","w","u","v"].slice(0,s),p="."+d[s-1],c=d.map(e=>"int "+e),f=yX("sourceLocR",s-1).concat("inIdx.r"),m=yX("sourceLocG",s-1).concat("inIdx.g"),g=yX("sourceLocB",s-1).concat("inIdx.b"),x=yX("sourceLocA",s-1).concat("inIdx.a"),y="max"===r?"greaterThan":"lessThan",b=n?"":`
          inIdx = round(vec4(getBestIndicesAChannel(${f.join()}),
                             getBestIndicesAChannel(${m.join()}),
                             getBestIndicesAChannel(${g.join()}),
                             getBestIndicesAChannel(${x.join()})));`,v=`vec4(
            getAChannel(${f.join()}),
            hasNextCol ? getAChannel(${m.join()}) : 0.,
            hasNextRow ? getAChannel(${g.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${x.join()}) : 0.)`,w=n?"":`
      float getBestIndicesAChannel(${c.join()}) {
        return getChannel(getBestIndicesA(${d.join()}),
                                          vec2(${d.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${c.join()}) {
        return getChannel(getA(${d.join()}),
                               vec2(${d.slice(-2).join()}));
      }
      ${w}
      void main() {
        ${u} coords = getOutputCoords();
        bool hasNextCol = ${h[l-1]} < ${o[l-1]-1};
        bool hasNextRow = ${h[l-2]} < ${o[l-2]-1};
        ${a}
        ivec4 srcIdx = ivec4(sourceLocR${p}, sourceLocG${p},
          sourceLocB${p}, sourceLocA${p}) * ${t};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${v};

        for (int i = 0; i < ${t}; i++) {
          inIdx = srcIdx;
          ${b}
          vec4 candidate = ${v};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${y}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}}function b9(e,t,r,n){let a=[r];if(tN.assertAxesAreInnerMostDims("arg"+n.charAt(0).toUpperCase()+n.slice(1),a,t.shape.length),!(0,nN.env)().getBool("WEBGL_PACK_REDUCE")||t.shape.length<=2){let r=[],s=e.texData.get(t.dataId),i=null!==s&&s.isPacked,o=t;i&&r.push(o=e.unpackTensor(t));let[l,u]=tN.computeOutAndReduceShapes(o.shape,a),h=bD({inputs:{x:o},backend:e,attrs:{shape:[-1,nR.util.sizeFromShape(u)]}});r.push(h);let d=function e(t,r,n,a=null){let s=r.shape[0],i=r.shape[1];null!=a&&(s=a.shape[0],i=a.shape[1]);let o=tN.computeOptimalWindowSize(i),l=new b8({windowSize:o,inSize:i,batchSize:s,outSize:Math.ceil(i/o)},n,null==a),u=[r];null!=a&&u.push(a);let h=t.runWebGLProgram(l,u,"int32");if(1===h.shape[1])return h;let d=e(t,r,n,h);return t.disposeIntermediateTensorInfo(h),d}(e,h,n);r.push(d);let p=bD({inputs:{x:d},backend:e,attrs:{shape:l}});return r.forEach(t=>e.disposeIntermediateTensorInfo(t)),p}return function e(t,r,n,a=null){let s=null!=a?a.shape:r.shape,i=s[s.length-1],o=new b7(s,tN.computeOptimalWindowSize(i),n,null==a),l=null==a?[r]:[r,a],u=t.runWebGLProgram(o,l,"int32");if(u.shape.length===r.shape.length){let a=e(t,r,n,u);return t.disposeIntermediateTensorInfo(u),a}return u}(e,t,n)}let ve={kernelName:x.ArgMax,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s}=n,i=nR.util.parseAxisParam(s,a.shape),o=tN.getAxesPermutation(i,a.shape.length),l=a,u=[];null!=o&&(u.push(l=bU({inputs:{x:a},backend:r,attrs:{perm:o}})),i=tN.getInnerMostAxes(i.length,l.shape.length)),tN.assertAxesAreInnerMostDims("argMax",[i[0]],l.shape.length);let h=b9(r,l,i[0],"max");return u.forEach(e=>r.disposeIntermediateTensorInfo(e)),h}};var tN=tN;let vt={kernelName:x.ArgMin,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s}=n,i=nR.util.parseAxisParam(s,a.shape),o=tN.getAxesPermutation(i,a.shape.length),l=a,u=[];null!=o&&(u.push(l=bU({inputs:{x:a},backend:r,attrs:{perm:o}})),i=tN.getInnerMostAxes(i.length,l.shape.length)),tN.assertAxesAreInnerMostDims("argMin",[i[0]],l.shape.length);let h=b9(r,l,i[0],"min");return u.forEach(e=>r.disposeIntermediateTensorInfo(e)),h}},vr=bS({opSnippet:y2+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`}),vn={kernelName:x.Asin,backendName:"webgl",kernelFunc:vr},va=bS({opSnippet:y2+"return log(x + sqrt(x * x + 1.0));"}),vs={kernelName:x.Asinh,backendName:"webgl",kernelFunc:va},vi=bS({opSnippet:y2+`
  return atan(x);
`}),vo={kernelName:x.Atan,backendName:"webgl",kernelFunc:vi},vl=bT({opSnippet:bh+`
  return atan(a, b);
`,packedOpSnippet:`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+bp+`
  return result;
`}),vu={kernelName:x.Atan2,backendName:"webgl",kernelFunc:vl},vh=bS({opSnippet:y2+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`}),vd={kernelName:x.Atanh,backendName:"webgl",kernelFunc:vh};var tN=tN;class vp{constructor(e,t,r,n=!1,a=!1){if(this.variableNames=["x"],"avg"===t&&r)throw Error("Cannot compute positions for average pool.");const s=e.filterWidth,i=e.strideHeight,o=e.strideWidth,l=e.dilationHeight,u=e.dilationWidth,h=e.effectiveFilterHeight,d=e.effectiveFilterWidth,p=e.padInfo.top,c=e.padInfo.left;this.outputShape=e.outShape;const f="avg"===t,m=`((batch  * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + d`,g=`(xR * ${e.inWidth} + xC) * ${e.inChannels} + d`;let x="0.0";if(f||(x="-1.0 / 1e-20"),r){this.userCode=`
        const ivec2 strides = ivec2(${i}, ${o});
        const ivec2 pads = ivec2(${p}, ${c});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${h};
              wR += ${l}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${d};
                wC += ${u}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value >= currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${n?a?m:g:`wR * ${d} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}let y=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;"avg"===t&&(y="avgValue / max(count, 1.0)");const b=4*Math.floor(s/4),v=s%4,w=`
      if (${f}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = max(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${i}, ${o});
      const ivec2 pads = ivec2(${p}, ${c});
      const float initializationValue = ${x};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${x});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${h};
            wR += ${l}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${b}; wC += 4) {
            int xC = xCCorner + wC * ${u};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              getValue(batch, xR, xC + 3 * ${u}, d)
            );

            ${w}
          }

          int xC = xCCorner + ${b};
          if (${1===v}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${w}
          } else if (${2===v}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              initializationValue,
              initializationValue
            );

            ${w}
          } else if (${3===v}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              initializationValue
            );

            ${w}
          }
        }
        setOutput(${y});
      }
    `}}class vc{constructor(e,t,r,n=!1,a=!1){if(this.variableNames=["x"],"avg"===t&&r)throw Error("Cannot compute positions for average pool.");const s=e.filterWidth,i=e.strideDepth,o=e.strideHeight,l=e.strideWidth,u=e.dilationDepth,h=e.dilationHeight,d=e.dilationWidth,p=e.effectiveFilterDepth,c=e.effectiveFilterHeight,f=e.effectiveFilterWidth,m=e.padInfo.front,g=e.padInfo.top,x=e.padInfo.left;this.outputShape=e.outShape;const y="avg"===t;let b="0.0";if(y||(b="-1.0 / 1e-20"),r){this.userCode=`
        const ivec3 strides =
            ivec3(${i}, ${o}, ${l});
        const ivec3 pads = ivec3(${m}, ${g}, ${x});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${p};
              wD += ${u}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${e.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${c};
                wR += ${h}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${f};
                  wC += ${d}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value >= currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${n?a?`(((batch * ${e.inDepth} + xD) * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`((xD * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`wD * ${c} * ${f} +
                      wR * ${f} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}let v=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;"avg"===t&&(v="avgValue / max(count, 1.0)");const w=4*Math.floor(s/4),C=s%4,I=`
      if (${y}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = max(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${i}, ${o}, ${l});
      const ivec3 pads = ivec3(${m}, ${g}, ${x});
      const float initializationValue = ${b};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${b});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${p};
            wD += ${u}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${c};
            wR += ${h}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${w}; wC += 4) {
              int xC = xCCorner + wC * ${d};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                getValue(batch, xD, xR, xC + 3 * ${d}, ch)
              );

              ${I}
            }

            int xC = xCCorner + ${w};
            if (${1===C}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${I}
            } else if (${2===C}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                initializationValue,
                initializationValue
              );

              ${I}
            } else if (${3===C}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                initializationValue
              );

              ${I}
            }
          }
        }
        setOutput(${v});
      }
    `}}let vf={kernelName:x.AvgPool,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t;xl(a,"avgPool");let{filterSize:s,strides:i,pad:o,dimRoundingMode:l}=n;nR.util.assert(tN.eitherStridesOrDilationsAreOne(i,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${i} and dilations '1'`);let u=tN.computePool2DInfo(a.shape,s,i,1,o,l);if(1===u.filterWidth&&1===u.filterHeight&&nR.util.arraysEqual(u.inShape,u.outShape))return bf({inputs:{x:a},backend:r});let h=new vp(u,"avg",!1);return r.runWebGLProgram(h,[a],"float32")}};var tN=tN;let vm={kernelName:x.AvgPool3D,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{filterSize:s,strides:i,pad:o,dimRoundingMode:l,dataFormat:u}=n,h=new vc(tN.computePool3DInfo(a.shape,s,i,[1,1,1],o,l,u),"avg",!1);return r.runWebGLProgram(h,[a],"float32")}};var tN=tN;class vg{constructor(e){this.variableNames=["dy"],this.outputShape=e.inShape;const t=e.filterHeight,r=e.filterWidth,n=e.strideHeight,a=e.strideWidth,s=e.dilationHeight,i=e.dilationWidth,o=e.effectiveFilterHeight,l=e.effectiveFilterWidth,u=o-1-e.padInfo.top,h=l-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${u}, ${h});
      const float avgMultiplier = float(${1/(t*r)});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${o};
            wR += ${s}) {
          float dyR = float(dyRCorner + wR) / ${n}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${l};
            wC+= ${i}) {
            float dyC = float(dyCCorner + wC) / ${a}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}}class vx{constructor(e){this.variableNames=["dy"],this.outputShape=e.inShape;const t=e.filterDepth,r=e.filterHeight,n=e.filterWidth,a=e.strideDepth,s=e.strideHeight,i=e.strideWidth,o=e.dilationDepth,l=e.dilationHeight,u=e.dilationWidth,h=e.effectiveFilterDepth,d=e.effectiveFilterHeight,p=e.effectiveFilterWidth,c=h-1-e.padInfo.front,f=d-1-e.padInfo.top,m=p-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${c}, ${f}, ${m});
      const float avgMultiplier = float(${1/(t*r*n)});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${h};
            wD += ${o}) {
          float dyD = float(dyDCorner + wD) / ${a}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${d};
              wR += ${l}) {
            float dyR = float(dyRCorner + wR) / ${s}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${p};
                wC += ${u}) {
              float dyC = float(dyCCorner + wC) / ${i}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}let vy={kernelName:x.AvgPool3DGrad,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,input:s}=t,{filterSize:i,strides:o,pad:l,dimRoundingMode:u}=n,h=new vx(tN.computePool3DInfo(s.shape,i,o,[1,1,1],l,u));return r.runWebGLProgram(h,[a],s.dtype)}};var tN=tN;let vb={kernelName:x.AvgPoolGrad,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,input:s}=t;xl([a,s],"avgPoolGrad");let{filterSize:i,strides:o,pad:l}=n,u=new vg(tN.computePool2DInfo(s.shape,i,o,1,l));return r.runWebGLProgram(u,[a],s.dtype)}},vv={kernelName:x.BatchMatMul,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{a,b:s}=t,{transposeA:i,transposeB:o}=n;return bH({a,b:s,transposeA:i,transposeB:o,backend:r})}};var tN=tN;class vw{constructor(e,t,r,n,a,s){this.outputShape=[],this.variableNames=["x","mean","variance"],tN.assertAndGetBroadcastShape(e,t),tN.assertAndGetBroadcastShape(e,r);let i="0.0";null!=n&&(tN.assertAndGetBroadcastShape(e,n),this.variableNames.push("offset"),i="getOffsetAtOutCoords()");let o="1.0";null!=a&&(tN.assertAndGetBroadcastShape(e,a),this.variableNames.push("scale"),o="getScaleAtOutCoords()"),this.outputShape=e,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${i};
        float scale = ${o};
        float inv = scale * inversesqrt(variance + float(${s}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}}var tN=tN;class vC{constructor(e,t,r,n,a,s){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=["x","mean","variance"],tN.assertAndGetBroadcastShape(e,t),tN.assertAndGetBroadcastShape(e,r);let i="vec4(0.0)";null!=n&&(tN.assertAndGetBroadcastShape(e,n),this.variableNames.push("offset"),i="getOffsetAtOutCoords()");let o="vec4(1.0)";null!=a&&(tN.assertAndGetBroadcastShape(e,a),this.variableNames.push("scale"),o="getScaleAtOutCoords()"),this.outputShape=e,this.userCode=`
      void main() {
        vec4 offset = ${i};
        vec4 scale = ${o};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${s}));

        setOutput((x - mean) * inv + offset);
      }
    `}}let vI={kernelName:x.FusedBatchNorm,backendName:"webgl",kernelFunc:({inputs:e,backend:t,attrs:r})=>{let{x:n,mean:a,variance:s,offset:i,scale:o}=e;nR.util.assert(a.shape.length===s.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),nR.util.assert(null==i||a.shape.length===i.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),nR.util.assert(null==o||a.shape.length===o.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");let{varianceEpsilon:l}=r;null==l&&(l=.001);let u=[n,a,s],h=null;null!=i&&(h=i.shape,u.push(i));let d=null;null!=o&&(d=o.shape,u.push(o));let p=(0,nN.env)().getBool("WEBGL_PACK_NORMALIZATION")?new vC(n.shape,a.shape,s.shape,h,d,l):new vw(n.shape,a.shape,s.shape,h,d,l);return t.runWebGLProgram(p,u,u[0].dtype)}};var tN=tN,pB=tJ;class vk{constructor(e){let t;this.variableNames=["source"],this.outputShape=e,this.rank=e.length;const r=xS(this.rank);this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const n=function(e){if(1===e)return"sourceLoc";if(e<=6)return vS.slice(0,e).map(e=>"sourceLoc."+e).join(",");throw Error(`Slicing for rank ${e} is not yet supported`)}(this.rank),a=e.map((e,t)=>`sourceLoc.${vS[t]} = start[${t}] + coords.${vS[t]};`);t=`
        ${r} sourceLoc;
        ${r} coords = getOutputCoords();
        ${a.join("\n")}
      `,this.userCode=`
      void main() {
        ${t}
        setOutput(getSource(${n}));
      }
    `}}let vS=["x","y","z","w","u","v"];class vT{constructor(e){this.variableNames=["source"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const t=xS(this.rank),r=yX("coords",this.rank),n=yX("sourceLoc",this.rank),a=1===this.rank?"sourceLoc":`vec2(${n.slice(-2).join()})`,s=`getChannel(getSource(${n.join()}), ${a})`,i=`
      result.x = ${s};
      if (++${r[this.rank-1]} < ${e[this.rank-1]}) {
        ++${n[this.rank-1]};
        result.y = ${s};
        --${n[this.rank-1]};
      }
    `,o=1===this.rank?"":`
      --${r[this.rank-1]};
      if (++${r[this.rank-2]} < ${e[this.rank-2]}) {
        ++${n[this.rank-2]};
        result.z = ${s};
        if (++${r[this.rank-1]} < ${e[this.rank-1]}) {
          ++${n[this.rank-1]};
          result.w = ${s};
        }
      }
    `,l=this.rank<=4?`sourceLoc = coords +
            ${t}(${e.map((e,t)=>`start[${t}]`).join()});`:e.map((e,t)=>`${n[t]} = ${r[t]} + start[${t}];`).join("\n");this.userCode=`
      void main() {
        ${t} coords = getOutputCoords();
        ${t} sourceLoc;
        ${l}
        vec4 result = vec4(0.);
        ${i}
        ${o}
        setOutput(result);
      }
    `}}function vN(e){let t,r,n,a,s,{inputs:i,backend:o,attrs:l}=e,{x:u}=i,{begin:h,size:d}=l,[p,c]=pB.parseSliceParams(u,h,d);if(pB.assertParamsValid(u,p,c),0===nR.util.sizeFromShape(c))return o.makeTensorInfo(c,u.dtype,[]);if(o.shouldExecuteOnCPU([u])||"string"===u.dtype){let e=yF(o.texData.get(u.dataId).values,p,c,u.shape,u.dtype);return o.makeTensorInfo(c,u.dtype,e)}let{isPacked:f}=o.texData.get(u.dataId),m=pB.isSliceContinous(u.shape,p,c);if(f||!m){let e=(0,nN.env)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new vT(c):new vk(c),t=[p];return o.runWebGLProgram(e,[u],u.dtype,t)}return o.uploadToGPU(u.dataId),t=o.texData.get(u.dataId),r=o.makeTensorInfo(c,u.dtype),Object.assign(n=o.texData.get(r.dataId),t),n.refCount=1,n.shape=c,n.dtype=u.dtype,a=pB.computeFlatOffset(p,nR.util.computeStrides(u.shape)),t.slice&&(a+=t.slice.flatOffset),n.slice={flatOffset:a,origDataId:t.slice&&t.slice.origDataId||u.dataId},s=o.dataRefCount.get(n.slice.origDataId)||1,o.dataRefCount.set(n.slice.origDataId,s+1),r}let v$={kernelName:x.Slice,backendName:"webgl",kernelFunc:vN},vR={kernelName:x.BatchToSpaceND,backendName:"webgl",kernelFunc:e=>{let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{blockShape:s,crops:i}=n;nR.util.assert(a.shape.length<=4,()=>"batchToSpaceND for rank > 4 with a WebGL backend not implemented yet");let o=s.reduce((e,t)=>e*t),l=tN.getReshaped(a.shape,s,o),u=tN.getPermuted(l.length,s.length),h=tN.getReshapedPermuted(a.shape,s,o),d=tN.getSliceBeginCoords(i,s.length),p=tN.getSliceSize(h,i,s.length),c=[],f=bD({inputs:{x:a},backend:r,attrs:{shape:l}}),m=bU({inputs:{x:f},backend:r,attrs:{perm:u}}),g=bD({inputs:{x:m},backend:r,attrs:{shape:h}}),x=vN({inputs:{x:g},backend:r,attrs:{begin:d,size:p}});return c.push(f),c.push(m),c.push(g),c.forEach(e=>r.disposeIntermediateTensorInfo(e)),x}},vA={kernelName:x.Bincount,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,weights:s}=t,{size:i}=n,o=x9(r.readSync(a.dataId),r.readSync(s.dataId),s.dtype,s.shape,i);return r.makeTensorInfo([i],s.dtype,o)}},vE=`
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`,vF=`
  return float(int(a.r) & int(b.r));
`,vD={kernelName:x.BitwiseAnd,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n}=e,{a,b:s}=r,i=(0,nN.env)().getBool("WEBGL_PACK_BINARY_OPERATIONS"),o=(0,nN.env)().getNumber("WEBGL_VERSION");if(n.shouldExecuteOnCPU([a,s])||1===o){let e=n.texData.get(a.dataId).values,t=n.texData.get(s.dataId).values,[r,i]=yt(a.shape,s.shape,e,t,a.dtype),o=n.makeTensorInfo(i,a.dtype);return n.texData.get(o.dataId).values=r,o}return t=i?new bc(vE,a.shape,s.shape,!1):new bd(vF,a.shape,s.shape),n.runWebGLProgram(t,[a,s],a.dtype)}};var tN=tN;let vO={kernelName:x.BroadcastArgs,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{s0:n,s1:a}=t,s=r.readSync(n.dataId),i=r.readSync(a.dataId),o=tN.assertAndGetBroadcastShape(Array.from(s),Array.from(i));return r.makeTensorInfo([o.length],"int32",Int32Array.from(o))}},vL=bT({opSnippet:"return float(a != b);",cpuKernelImpl:yC,dtype:"bool"}),vz={kernelName:x.NotEqual,backendName:"webgl",kernelFunc:vL};function v_(e){let{inputs:t,backend:r}=e,{input:n}=t;return bf({inputs:{x:r.texData.get(n.dataId).complexTensorInfos.real},backend:r})}let vM={kernelName:x.Real,backendName:"webgl",kernelFunc:v_},vP={kernelName:x.Cast,backendName:"webgl",kernelFunc:function e(t){let{inputs:r,backend:n,attrs:a}=t,{x:s}=r,{dtype:i}=a;if("complex64"===i){if("complex64"===s.dtype)return bf({inputs:{x:s},backend:n});let t=tx.zeros(s.shape),r=e({inputs:{x:s},backend:n,attrs:{dtype:"float32"}}),a=bg({inputs:{real:r,imag:t},backend:n});return t.dispose(),n.disposeIntermediateTensorInfo(r),a}if("complex64"===s.dtype){let t=v_({inputs:{input:s},backend:n}),r=e({inputs:{x:t},backend:n,attrs:{dtype:i}});return n.disposeIntermediateTensorInfo(t),r}if(!nR.util.hasEncodingLoss(s.dtype,i)){let e=bf({inputs:{x:s},backend:n});return{dataId:e.dataId,shape:e.shape,dtype:i}}if(n.shouldExecuteOnCPU([s])){let[e,t,r]=yr(n.texData.get(s.dataId).values,s.shape,s.dtype,i);return n.makeTensorInfo(e,t,r)}if("int32"===i){let e,t;return e=new y1(s.shape,"return float(int(x));"),{dataId:(t=n.runWebGLProgram(e,[s],"int32")).dataId,shape:t.shape,dtype:t.dtype}}if("bool"===i){let e=n.makeTensorInfo([],"bool",nR.util.getTypedArrayFromDType("bool",1)),t=vL({inputs:{a:s,b:e},backend:n});return n.disposeIntermediateTensorInfo(e),t}throw Error(`Error in Cast: failed to cast ${s.dtype} to ${i}`)}},vB="return ceil(x);",vW=bS({opSnippet:vB,packedOpSnippet:vB,cpuKernelImpl:yn}),vG={kernelName:x.Ceil,backendName:"webgl",kernelFunc:vW};class vU{constructor(e){this.variableNames=["A"],this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=e,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}}class vV{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=e,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}}let vH={kernelName:x.ClipByValue,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{clipValueMin:i,clipValueMax:o}=a;return t=(0,nN.env)().getBool("WEBGL_PACK_CLIP")?new vV(s.shape):new vU(s.shape),n.runWebGLProgram(t,[s],s.dtype,[[i],[o]])}};class vq{constructor(e){this.variableNames=["real","imag"],this.outputShape=e,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}}function vj(e,t){return{dataId:t.dataId,dtype:t.dtype,shape:e.shape}}let vX={kernelName:x.ComplexAbs,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{x:n}=t,a=r.texData.get(n.dataId),s=new vq(n.shape),i=[vj(n,a.complexTensorInfos.real),vj(n,a.complexTensorInfos.imag)];return r.runWebGLProgram(s,i,i[0].dtype)}};var tN=tN,tN=tN,tN=tN;class vK{constructor(e){this.outputShape=[],this.outputShape=tN.computeOutShape(e,1),this.variableNames=e.map((e,t)=>`T${t}`);const t=Array(e.length-1);t[0]=e[0][1];for(let r=1;r<t.length;r++)t[r]=t[r-1]+e[r][1];const r=[`if (yC < ${t[0]}) setOutput(getT0(yR, yC));`];for(let e=1;e<t.length;e++){const n=t[e-1];r.push(`else if (yC < ${t[e]}) setOutput(getT${e}(yR, yC-${n}));`)}const n=t.length,a=t[t.length-1];r.push(`else setOutput(getT${n}(yR, yC-${a}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${r.join("\n        ")}
      }
    `}}var tN=tN;class vY{constructor(e,t){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=tN.computeOutShape(e,t);const r=this.outputShape,n=r.length,a=xS(n),s=yX("coords",n),i=["x","y","z","w","u","v"].slice(0,n);this.variableNames=e.map((e,t)=>`T${t}`);const o=Array(e.length-1);o[0]=e[0][t];for(let r=1;r<o.length;r++)o[r]=o[r-1]+e[r][t];const l=i[t],u=i.slice(-2),h=i.join();let d=`if (${l} < ${o[0]}) {
        return getChannel(
            getT0(${h}), vec2(${u.join()}));
        }`;for(let e=1;e<o.length;e++){const t=o[e-1];d+=`
        if (${l} < ${o[e]}  && ${l} >= ${o[e-1]}) {
          return getChannel(
            getT${e}(${vZ(i,l,t)}),
            vec2(${vZ(u,l,t)}));
        }`}const p=o.length,c=o[o.length-1];d+=`
        return getChannel(
          getT${p}(${vZ(i,l,c)}),
          vec2(${vZ(u,l,c)}));`,this.userCode=`
      float getValue(${i.map(e=>"int "+e)}) {
        ${d}
      }

      void main() {
        ${a} coords = getOutputCoords();
        vec4 result = vec4(getValue(${s}), 0., 0., 0.);

        ${s[n-1]} = ${s[n-1]} + 1;
        if (${s[n-1]} < ${r[n-1]}) {
          result.g = getValue(${s});
        }

        ${s[n-2]} = ${s[n-2]} + 1;
        if (${s[n-2]} < ${r[n-2]}) {
          result.a = getValue(${s});
        }

        ${s[n-1]} = ${s[n-1]} - 1;
        if (${s[n-2]} < ${r[n-2]} &&
            ${s[n-1]} < ${r[n-1]}) {
          result.b = getValue(${s});
        }
        setOutput(result);
      }
    `}}function vZ(e,t,r){let n=e.indexOf(t);return e.map((e,t)=>t===n?`${e} - ${r}`:e).join()}function vJ(e){let{inputs:t,backend:r}=e,{input:n}=t;return bf({inputs:{x:r.texData.get(n.dataId).complexTensorInfos.imag},backend:r})}let vQ={kernelName:x.Imag,backendName:"webgl",kernelFunc:vJ};function v0(e){let{inputs:t,backend:r,attrs:n}=e,{axis:a}=n,s=nR.util.parseAxisParam(a,t[0].shape)[0],i=t.map(e=>e.shape);tN.assertParamsConsistent(i,s);let o=tN.computeOutShape(t.map(e=>e.shape),s);if(0===nR.util.sizeFromShape(o))return r.makeTensorInfo(o,t[0].dtype,[]);let l=t.filter(e=>nR.util.sizeFromShape(e.shape)>0);return 1===l.length?bf({inputs:{x:l[0]},backend:r}):function e(t,r,n){var a,s,i;let o,l=t[0].dtype;if("complex64"===l){let a=t.map(e=>v_({inputs:{input:e},backend:n})),s=t.map(e=>vJ({inputs:{input:e},backend:n})),i=e(a,r,n),o=e(s,r,n),l=bg({inputs:{real:i,imag:o},backend:n});return a.forEach(e=>n.disposeIntermediateTensorInfo(e)),s.forEach(e=>n.disposeIntermediateTensorInfo(e)),n.disposeIntermediateTensorInfo(i),n.disposeIntermediateTensorInfo(o),l}let u=n.shouldExecuteOnCPU(t);if("string"===l&&(u=!0),u){let e=t.map(e=>{let t=nR.util.sizeFromShape(e.shape.slice(r));return bD({inputs:{x:e},backend:n,attrs:{shape:[-1,t]}})}),a=ya(e.map(e=>({vals:n.readSync(e.dataId),shape:e.shape})),tN.computeOutShape(e.map(e=>e.shape),1),l,1===e[0].shape[0]),s=tN.computeOutShape(t.map(e=>e.shape),r),i=n.makeTensorInfo(s,l,a);return e.forEach(e=>n.disposeIntermediateTensorInfo(e)),i}let h=t.filter(e=>nR.util.sizeFromShape(e.shape)>0),d=(0,nN.env)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")&&h[0].shape.length>1;if(1===h.length){let e=d?new y1(t[0].shape,y6):new be(t[0].shape,y6);return n.runWebGLProgram(e,t,l)}let p=(0,nN.env)().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER");if(h.length>p){let t=[];for(let a=0;a<h.length;a+=p){let s=h.slice(a,a+p);t.push(e(s,r,n))}let a=e(t,r,n);for(let e of t)n.disposeIntermediateTensorInfo(e);return a}if(d){let e=new vY(h.map(e=>e.shape),r);return n.runWebGLProgram(e,h,l)}let{tensors2D:c,outShape:f}=(a=h,s=r,i=n,o=tN.computeOutShape(a.map(e=>e.shape),s),{tensors2D:a.map(e=>bD({inputs:{x:e},attrs:{shape:[-1,nR.util.sizeFromShape(e.shape.slice(s))]},backend:i})),outShape:o}),m=new vK(c.map(e=>e.shape)),g=n.runWebGLProgram(m,c,l);c.forEach(e=>n.disposeIntermediateTensorInfo(e));let x=bD({inputs:{x:g},attrs:{shape:f},backend:n});return n.disposeIntermediateTensorInfo(g),x}(l,s,r)}let v1={kernelName:x.Concat,backendName:"webgl",kernelFunc:v0};var tN=tN;class v2{constructor(e,t=!1,r=null,n=!1,a=!1){this.variableNames=["x","W"],this.outputShape=e.outShape;const s=e.padInfo.top,i=e.padInfo.left,o=e.strideHeight,l=e.strideWidth,u=e.dilationHeight,h=e.dilationWidth,d=e.filterHeight,p=e.filterWidth,c=4*Math.floor(e.inChannels/4),f=e.inChannels%4,m="channelsLast"===e.dataFormat;let g="",x="";r&&(g=n?`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${r}
        }`:a?`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${r}
        }`:`
          float activation(float x) {
            ${r}
          }
        `,x="result = activation(result);"),t&&this.variableNames.push("bias"),n&&this.variableNames.push("preluActivationWeights"),a&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${g}

      const ivec2 strides = ivec2(${o}, ${l});
      const ivec2 pads = ivec2(${s}, ${i});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${m?3:1}];

        ivec2 xRCCorner =
            ivec2(coords[${m?1:2}], coords[${m?2:3}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${d}; wR++) {
          int xR = xRCorner + wR * ${u};

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${p}; wC++) {
            int xC = xCCorner + wC * ${h};

            if (xC < 0 || xC >= ${e.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${c}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${m}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${1===f}) {

              if (${m}) {
                dotProd +=
                    getX(batch, xR, xC, ${c}) *
                    getW(wR, wC, ${c}, d2);
              } else {
                dotProd +=
                    getX(batch, ${c}, xR, xC) *
                    getW(wR, wC, ${c}, d2);
              }

            } else if (${2===f}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${c}, d2),
                getW(wR, wC, ${c} + 1, d2)
              );

              if (${m}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${c}),
                  getX(batch, xR, xC, ${c} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${c}, xR, xC),
                  getX(batch, ${c} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${3===f}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${c}, d2),
                getW(wR, wC, ${c} + 1, d2),
                getW(wR, wC, ${c} + 2, d2)
              );

              if (${m}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${c}),
                  getX(batch, xR, xC, ${c} + 1),
                  getX(batch, xR, xC, ${c} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${c}, xR, xC),
                  getX(batch, ${c} + 1, xR, xC),
                  getX(batch, ${c} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${t?"result += getBiasAtOutCoords();":""}
        ${x}
        setOutput(result);
      }
    `}}class v3{constructor(e){this.variableNames=["x","W"],this.outputShape=e.outShape;const t=e.padInfo.front,r=e.padInfo.top,n=e.padInfo.left,a=e.strideDepth,s=e.strideHeight,i=e.strideWidth,o=e.dilationDepth,l=e.dilationHeight,u=e.dilationWidth,h=e.filterDepth,d=e.filterHeight,p=e.filterWidth,c=4*Math.floor(e.inChannels/4),f=e.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${a}, ${s}, ${i});
      const ivec3 pads = ivec3(${t}, ${r}, ${n});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${h}; wF++) {
          int xF = xFCorner + wF * ${o};

          if (xF < 0 || xF >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${d}; wR++) {
            int xR = xRCorner + wR * ${l};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${p}; wC++) {
              int xC = xCCorner + wC * ${u};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${c}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${1===f}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${c}) *
                  getW(wF, wR, wC, ${c}, d2);
              } else if (${2===f}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${c}),
                  getX(batch, xF, xR, xC, ${c} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${c}, d2),
                  getW(wF, wR, wC, ${c} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${3===f}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${c}),
                  getX(batch, xF, xR, xC, ${c} + 1),
                  getX(batch, xF, xR, xC, ${c} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${c}, d2),
                  getW(wF, wR, wC, ${c} + 1, d2),
                  getW(wF, wR, wC, ${c} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class v4{constructor(e,t=!1,r=null,n=!1,a=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=xE(this.outputShape.length);const s=e.padInfo.left,i=e.strideWidth,o=e.dilationWidth,l=e.filterHeight,u=e.filterWidth;let h=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let e=0;e<u;e++)h+=`
           vec4 xTexelC${2*e};
           int xTexelC${2*e}Ready;
           vec4 xTexelC${2*e+1};
           int xTexelC${2*e+1}Ready;
           vec4 xC${e};`;h+=`
     for (int r = 0; r < ${l}; r++) {
      for (int d1 = 0; d1 < ${e.inChannels}; d1 += 2) {
       `;for(let e=0;e<u;e++)h+=`
           xTexelC${2*e} = vec4(0.0);
           xTexelC${2*e}Ready = 0;
           xTexelC${2*e+1} = vec4(0.0);
           xTexelC${2*e+1}Ready = 0;
           xC${e} = vec4(0.0);`;h+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let t=0;t<(u+1)/2;t++){const r=2*t;if(h+=`
           xC = xCCorner + ${r*o};
           `,1===i){if(r<u&&(s%2==1?(h+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${r}Ready == 0) {
                   xTexelC${r} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${r}.zw = vec2(0.0);
                   }
                   xTexelC${r}Ready = 1;
                 }
               `,1===o&&r>0?h+=`
                 xC${r} = vec4(xTexelC${r-2}.zw, xTexelC${r}.xy);
                 `:h+=`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${r} = vec4(previous.zw, xTexelC${r}.xy);
                   } else {
                     xC${r} = vec4(0.0, 0.0, xTexelC${r}.xy);
                   }
                   `):h+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${r}Ready == 0) {
                   xTexelC${r} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${r}.zw = vec2(0.0);
                   }
                   xTexelC${r}Ready = 1;
                 }

                 xC${r} = xTexelC${r};
                 `,r+1<u)){const e=s%2==0?nR.util.nearestLargerEven(o):o;o%2==0&&s%2==1||o%2!=0&&s%2!=1?(h+=`
                   xCOffset = xC + imod(pads[1], 2) + ${e};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${r+1}Ready == 0) {
                     xTexelC${r+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${r+1}.zw = vec2(0.0);
                     }
                     xTexelC${r+1}Ready = 1;
                   }
                   `,o>1?h+=`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${r+1} = vec4(previous.zw, xTexelC${r+1}.xy);
                     } else {
                      xC${r+1} = vec4(0.0, 0.0, xTexelC${r+1}.xy);
                     }
                     `:h+=`
                     xC${r+1} = vec4(xTexelC${r}.zw, xTexelC${r+1}.xy);
                     `):1===e?h+=`
                     xC${r+1} = xTexelC${r};
                     `:h+=`
                     xCOffset = xC + ${e};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${r+1}Ready == 0) {
                       xTexelC${r+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${r+1}.zw = vec2(0.0);
                       }
                       xTexelC${r+1}Ready = 1;
                     }

                     xC${r+1} = xTexelC${r+1};
                     `}}else r<u&&(s%2==1?(h+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${r}Ready == 0) {
                   xTexelC${r} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${r}.zw = vec2(0.0);
                   }
                   xTexelC${r}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${r+1}Ready == 0) {
                   xTexelC${r+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${r+1}.zw = vec2(0.0);
                   }
                   xTexelC${r+1}Ready = 1;
                 }

                 xC${r} = vec4(xTexelC${r}.zw, xTexelC${r+1}.zw);
               `,r+1<u&&(h+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${r+1} = vec4(xTexelC${r+1}.xy, final.xy);
                 `)):(h+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${r}Ready == 0) {
                   xTexelC${r} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${r}.zw = vec2(0.0);
                   }
                   xTexelC${r}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${r+1}Ready == 0) {
                   xTexelC${r+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${r+1}.zw = vec2(0.);
                   }
                   xTexelC${r+1}Ready = 1;
                 }

                 xC${r} = vec4(
                   xTexelC${r}.xy, xTexelC${r+1}.xy);
               `,r+1<u&&(h+=`
                   xC${r+1} = vec4(xTexelC${r}.zw, xTexelC${r+1}.zw);
                 `)));r<u&&(h+=`
             wTexel = getW(r, ${r}, d1, d2);
             dotProd += xC${r}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${e.inChannels}) {
               dotProd += xC${r}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,r+1<u&&(h+=`
               wTexel = getW(r, ${r+1}, d1, d2);
               dotProd += xC${r+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${e.inChannels}) {
                 dotProd += xC${r+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}h+=`
     }
   
     }
   
     }
   `;let d="",p="";r&&(d=n?`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${r}
         }`:a?`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${r}
         }`:`vec4 activation(vec4 x) {
           ${r}
         }`,p="result = activation(result);"),t&&this.variableNames.push("bias"),n&&this.variableNames.push("preluActivationWeights"),a&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
       ${d}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${h}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${t?"result += getBiasAtOutCoords();":""}
         ${p}
         setOutput(result);
       }
     `}}class v5{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec4"},{name:"pad",type:"ivec2"},{name:"stride",type:"ivec2"},{name:"dilation",type:"ivec2"},{name:"inChannels",type:"int"},{name:"itemsPerBlockRow",type:"int"},{name:"outWidth",type:"int"}],this.outputShape=e,this.enableShapeUniforms=xE(this.outputShape.length);const{dataFormat:r}=t,n=xd(),a="channelsLast"===r,s=a?1:2,i=a?2:3,o=this.enableShapeUniforms?"if(blockIndex < outShape[2] && pos < outShape[1]) {":`if(blockIndex < ${e[2]} && pos < ${e[1]}) {`;let l="";for(let e=0;e<=1;e++)for(let t=0;t<=1;t++)l+=`
          blockIndex = rc.z + ${t};
          pos = rc.y + ${e};

          ${o}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${s}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${i}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${a}) {
                  innerDims = vec2(d1, ch);
                  result[${2*e+t}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${2*e+t}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${l}

        ${n.output} = result;
      }
    `}}function v6(e,t){let r=e.length;return r>=3?t?[...e.slice(0,-3),e[r-3]*e[r-2],e[r-1]]:[...e.slice(0,-3),e[r-3],e[r-2]*e[r-1]]:!t&&1===r&&e[0]>1?[e[0],1]:null}function v8({x:e,filter:t,convInfo:r,backend:n,bias:a=null,preluActivationWeights:s=null,leakyreluAlpha:i=0,activation:o=null}){let l,u=e.shape,h=n.texData.get(e.dataId),d=r.inChannels,p=u[0]*u[1]*u[2],c=r.outChannels,f="channelsLast"===r.dataFormat,m=[];if(null!=s){let e=v6(s.shape,f);null!=e&&(s=bD({inputs:{x:s},backend:n,attrs:{shape:e}}),m.push(s))}if(null!=a){let e=v6(a.shape,f);null!=e&&(a=bD({inputs:{x:a},backend:n,attrs:{shape:e}}),m.push(a))}if(!((1===p||1===c)&&d>1e3)&&h.isPacked&&f&&null!=h.texture&&u[2]%2!=0&&nR.util.arraysEqual(h.shape.slice(-3),u.slice(-3))){let d=u[0]*u[1]*(u[2]+1),p={dataId:e.dataId,shape:[1,d,r.inChannels],dtype:e.dtype},c=h.shape;h.shape=h.shape.slice(),h.shape[h.shape.length-2]++,nR.util.assert(g7(h.shape,p.shape),()=>`packed reshape ${h.shape} to ${p.shape} isn't free`);let f=bD({inputs:{x:t},backend:n,attrs:{shape:[1,r.inChannels,r.outChannels]}});m.push(f);let g=bH({a:p,b:f,backend:n,transposeA:!1,transposeB:!1,bias:a,activation:o,preluActivationWeights:s,leakyreluAlpha:i}),x=n.texData.get(g.dataId);nR.util.assert(x.isPacked,()=>"batchMatMul result is expected to be packed"),h.shape=c,x.shape=r.outShape,(l=bf({inputs:{x:g},backend:n})).shape=r.outShape,m.push(g)}else{let u=r.outHeight*r.outWidth,h=bD({inputs:{x:e},backend:n,attrs:{shape:f?[r.batchSize,u,r.inChannels]:[r.batchSize,r.inChannels,u]}}),d=bD({inputs:{x:t},backend:n,attrs:{shape:[1,r.inChannels,r.outChannels]}}),p=bH({a:f?h:d,b:f?d:h,transposeA:!f,transposeB:!1,backend:n,bias:a,activation:o,preluActivationWeights:s,leakyreluAlpha:i});l=bD({inputs:{x:p},backend:n,attrs:{shape:r.outShape}}),m.push(h),m.push(d),m.push(p)}for(let e of m)n.disposeIntermediateTensorInfo(e);return l}function v7({x:e,filter:t,convInfo:r,backend:n,bias:a=null,preluActivationWeights:s=null,leakyreluAlpha:i=0,activation:o=null}){let{filterWidth:l,filterHeight:u,inChannels:h,outWidth:d,outHeight:p,dataFormat:c}=r,f="channelsLast"===c,m=l*u*h,g=p*d,x=[r.batchSize,m,g],y=[];if(null!=s){let e=v6(s.shape,f);null!=e&&(s=bD({inputs:{x:s},backend:n,attrs:{shape:e}}),y.push(s))}if(null!=a){let e=v6(a.shape,f);null!=e&&(a=bD({inputs:{x:a},backend:n,attrs:{shape:e}}),y.push(a))}let b=bD({inputs:{x:t},backend:n,attrs:{shape:[1,m,nR.util.sizeFromShape(t.shape)/m]}});y.push(b);let v=new v5(x,r),w=[e.shape,[r.padInfo.top,r.padInfo.left],[r.strideHeight,r.strideWidth],[r.dilationHeight,r.dilationWidth],[r.inChannels],[r.filterWidth*r.inChannels],[r.outWidth]],C=n.runWebGLProgram(v,[e],"float32",w),I=bD({inputs:{x:C},backend:n,attrs:{shape:x}});y.push(C),y.push(I);let k=null!=a,S=null!=s,T="leakyrelu"===o,N=o?bN(o,!0):null,$=new b$(f?I.shape:b.shape,f?b.shape:I.shape,f?[r.batchSize,g,r.outChannels]:[r.batchSize,r.outChannels,g],!0,!1,k,N,S,T),R=f?[I,b]:[b,I];if(a&&R.push(a),S&&R.push(s),T){let e=n.makeTensorInfo([],"float32",nR.util.createScalarValue(i,"float32"));R.push(e),y.push(e)}let A=n.runWebGLProgram($,R,"float32"),E=bD({inputs:{x:A},backend:n,attrs:{shape:r.outShape}});for(let e of(y.push(A),y))n.disposeIntermediateTensorInfo(e);return E}let v9={kernelName:x.Conv2D,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s,filter:i}=r,{strides:o,pad:l,dataFormat:u,dilations:h,dimRoundingMode:d}=a,p=tN.convertConv2DDataFormat(u),c=tN.computeConv2DInfo(s.shape,i.shape,o,h,l,d,!1,p);if(1===c.filterHeight&&1===c.filterWidth&&1===c.dilationHeight&&1===c.dilationWidth&&1===c.strideHeight&&1===c.strideWidth&&("SAME"===c.padInfo.type||"VALID"===c.padInfo.type))t=v8({x:s,filter:i,convInfo:c,backend:n});else if(c.strideWidth<=2&&"channelsLast"===p&&(0,nN.env)().getBool("WEBGL_EXP_CONV")){let e=new v4(c),r=[[c.padInfo.top,c.padInfo.left],[c.strideHeight,c.strideWidth],[c.dilationHeight,c.dilationWidth],[c.inHeight,c.inWidth]];t=n.runWebGLProgram(e,[s,i],"float32",r)}else if((0,nN.env)().getBool("WEBGL_CONV_IM2COL"))t=v7({x:s,filter:i,convInfo:c,backend:n});else{let e=new v2(c);t=n.runWebGLProgram(e,[s,i],"float32")}let f=bD({inputs:{x:t},backend:n,attrs:{shape:c.outShape}});return n.disposeIntermediateTensorInfo(t),f}};var tN=tN;class we{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideHeight,r=e.strideWidth,n=e.padInfo.top,a=e.padInfo.left,s="channelsLast"===e.dataFormat;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${n};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${r} - ${a};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              ${s?`float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);`:`float dyValue = getDy(b, d2, yR, yC);
              float xValue = getX(b, d1, xR, xC);
              dotProd += (xValue * dyValue);`}
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class wt{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterHeight,r=e.filterWidth,n=e.strideHeight,a=e.strideWidth,s="channelsLast"===e.dataFormat,i=t-1-e.padInfo.top,o=r-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${i}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${s?3:1}];

        ivec2 dyCorner = ivec2(coords[${s?1:2}], coords[${s?2:3}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${n}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${r}; wC++) {
            float dyC = float(dyCCorner + wC) / ${a}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${r} - 1 - wC;

            for (int d2 = 0; d2 < ${e.outChannels}; d2++) {

              if (${s}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}}class wr{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideDepth,r=e.strideHeight,n=e.strideWidth,a=e.padInfo.front,s=e.padInfo.top,i=e.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yF = 0; yF < ${e.outDepth}; yF++) {
            int xF = wF + yF * ${t} - ${a};

            if (xF < 0 || xF >= ${e.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${e.outHeight}; yR++) {
              int xR = wR + yR * ${r} - ${s};

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${e.outWidth}; yC++) {
                int xC = wC + yC * ${n} - ${i};

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class wn{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterDepth,r=e.filterHeight,n=e.filterWidth,a=e.strideDepth,s=e.strideHeight,i=e.strideWidth,o=t-1-e.padInfo.front,l=r-1-e.padInfo.top,u=n-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${o}, ${l}, ${u});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${t}; wF++) {
          float dyF = float(dyFCorner + wF) / ${a}.0;

          if (dyF < 0.0 || dyF >= ${e.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${t} - 1 - wF;

          for (int wR = 0; wR < ${r}; wR++) {
            float dyR = float(dyRCorner + wR) / ${s}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${r} - 1 - wR;

            for (int wC = 0; wC < ${n}; wC++) {
              float dyC = float(dyCCorner + wC) / ${i}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${n} - 1 - wC;

              for (int d2 = 0; d2 < ${e.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}let wa={kernelName:x.Conv2DBackpropFilter,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,dy:s}=t,{strides:i,pad:o,dataFormat:l,dimRoundingMode:u,filterShape:h}=n,d=tN.convertConv2DDataFormat(l),p=new we(tN.computeConv2DInfo(a.shape,h,i,1,o,u,!1,d));return r.runWebGLProgram(p,[a,s],"float32")}};var tN=tN;class ws{constructor(e){this.variableNames=["dy","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"strides",type:"vec2"}],this.outputShape=e.inShape,this.enableShapeUniforms=xE(this.outputShape.length);const t=e.filterHeight,r=e.filterWidth,n=t-1-e.padInfo.top,a=r-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${n}, ${a});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];

        ivec2 dyCorner = ivec2(coords[1], coords[2]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        vec4 result = vec4(0.);
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / strides[0];
          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);
          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${r}; wC++) {
            int wCPerm = ${r} - 1 - wC;

            float dyC = float(dyCCorner + wC) / strides[1];
            bool idyCVal = (dyC >= 0.0) && (dyC < ${e.outWidth}.0)
              && (fract(dyC) == 0.0);
            int idyC = int(dyC);

            float dyC2 = float(dyCCorner + wC + 1) / strides[1];
            bool idyCVal2 = (dyC2 >= 0.0) && (dyC2 < ${e.outWidth}.0)
              && (fract(dyC2) == 0.0);
            int idyC2 = int(dyC2);

            if (idyCVal && idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec4 dySample2 = (idyC / 2 == idyC2 / 2) ?
                  dySample : getDy(batch, idyR, idyC2, d2);

                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));

                dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample2.xy : dySample2.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC2, d2);
                vec2 dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            }
          }
        }
        setOutput(result);
      }
    `}}let wi={kernelName:x.Conv2DBackpropInput,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,filter:s}=t,{inputShape:i,strides:o,pad:l,dataFormat:u,dimRoundingMode:h}=n,d=tN.convertConv2DDataFormat(u),p=tN.computeConv2DInfo(i,s.shape,o,1,l,h,!1,d);if((0,nN.env)().getBool("WEBGL_PACK_CONV2DTRANSPOSE")&&"channelsLast"===d){let e=[[p.strideHeight,p.strideWidth]],t=new ws(p);return r.runWebGLProgram(t,[a,s],"float32",e)}{let e=new wt(p);return r.runWebGLProgram(e,[a,s],"float32")}}};var tN=tN;let wo={kernelName:x.Conv3D,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,filter:s}=t,{strides:i,pad:o,dilations:l}=n,u=new v3(tN.computeConv3DInfo(a.shape,s.shape,i,l,o));return r.runWebGLProgram(u,[a,s],"float32")}};var tN=tN;let wl={kernelName:x.Conv3DBackpropFilterV2,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,dy:s}=t,{strides:i,pad:o,filterShape:l}=n,u=new wr(tN.computeConv3DInfo(a.shape,l,i,1,o));return r.runWebGLProgram(u,[a,s],"float32")}};var tN=tN;let wu={kernelName:x.Conv3DBackpropInputV2,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,filter:s}=t,{pad:i,strides:o,inputShape:l}=n,u=new wn(tN.computeConv3DInfo(l,s.shape,o,1,i));return r.runWebGLProgram(u,[a,s],"float32")}},wh=bS({opSnippet:bk+`
  return cos(x);
`,packedOpSnippet:`
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${bp}
  return result;
`}),wd={kernelName:x.Cos,backendName:"webgl",kernelFunc:wh},wp=bS({opSnippet:`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`}),wc={kernelName:x.Cosh,backendName:"webgl",kernelFunc:wp};class wf{constructor(e,t,r,n,a){this.variableNames=["Image","Boxes","BoxInd"],this.outputShape=[];const[s,i,o,l]=e,[u]=t,[h,d]=r;this.outputShape=[u,h,d,l];const[p,c]=[`${i-1}.0`,`${o-1}.0`],[f,m,g]=h>1?[`${(i-1)/(h-1)}`,"(y2-y1) * height_ratio",`y1*${p} + float(y)*(height_scale)`]:["0.0","0.0",`0.5 * (y1+y2) * ${p}`],[x,y,b]=d>1?[`${(o-1)/(d-1)}`,"(x2-x1) * width_ratio",`x1*${c} + float(x)*(width_scale)`]:["0.0","0.0",`0.5 * (x1+x2) * ${c}`];this.userCode=`
      const float height_ratio = float(${f});
      const float width_ratio = float(${x});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${s}) {
          return;
        }

        float height_scale = ${m};
        float width_scale = ${y};

        float in_y = ${g};
        if( in_y < 0.0 || in_y > ${p} ) {
          setOutput(float(${a}));
          return;
        }
        float in_x = ${b};
        if( in_x < 0.0 || in_x > ${c} ) {
          setOutput(float(${a}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${+("bilinear"===n)} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}}let wm={kernelName:x.CropAndResize,backendName:"webgl",kernelFunc:e=>{let{inputs:t,backend:r,attrs:n}=e,{image:a,boxes:s,boxInd:i}=t,{cropSize:o,method:l,extrapolationValue:u}=n,h=new wf(a.shape,s.shape,o,l,u);return r.runWebGLProgram(h,[a,s,i],"float32")}};(h=g||(g={})).Prod="*",h.Sum="+";class wg{constructor(e,t,r,n){this.op=e,this.outputShape=t,this.variableNames=["x"],this.customUniforms=[{name:"index",type:"float"}];const a=this.outputShape.length,s=this.op===g.Prod?"1.0":"0.0",i=r?s:`getX(${wx(a,"coords",this.op)})`,o=this.outputShape[this.outputShape.length-1];let l="",u="";r?(l=n?`end != ${o-1}`:"end != 0",u=n?"end + 1":"end - 1"):(l=n?`end + pow2 < ${o}`:"end >= pow2",u=n?"end + pow2":"end - pow2"),this.userCode=`
      void main() {
        ${xS(a)} coords = getOutputCoords();
        int end = ${wy(a,"coords",this.op)};
        float val = ${i};
        int pow2 = int(pow(2.0, index));
        if (${l}) {
          int idx = ${u};
          ${wy(a,"coords",this.op)} = idx;
          val ${this.op}= getX(${wx(a,"coords",this.op)});
        }
        setOutput(val);
      }
    `}}function wx(e,t,r){if(1===e)return`${t}`;if(2===e)return`${t}.x, ${t}.y`;if(3===e)return`${t}.x, ${t}.y, ${t}.z`;if(4===e)return`${t}.x, ${t}.y, ${t}.z, ${t}.w`;throw Error(`Cumulative ${r} for rank ${e} is not yet supported`)}function wy(e,t,r){if(1===e)return`${t}`;if(2===e)return`${t}.y`;if(3===e)return`${t}.z`;if(4===e)return`${t}.w`;throw Error(`Cumulative ${r} for rank ${e} is not yet supported`)}var tN=tN;function wb(e,t,r,n,a,s){let i=t.shape.length,o=tN.getAxesPermutation([n],i),l=t;null!=o&&(l=bU({inputs:{x:t},backend:r,attrs:{perm:o}}));let u=tN.getInnerMostAxes(1,i)[0];if(u!==i-1)throw Error(`WebGL cumprod shader expects an inner-most axis=${t.shape.length-1} but got axis=${n}`);let h=l.shape[u],d=bf({inputs:{x:l},backend:r});for(let t=0;t<=Math.ceil(Math.log2(h))-1;t++){let n=new wg(e,l.shape,!1,s),a=[[t]],i=d;d=r.runWebGLProgram(n,[d],d.dtype,a),r.disposeIntermediateTensorInfo(i)}if(a){let t=new wg(e,l.shape,a,s),n=d;d=r.runWebGLProgram(t,[d],d.dtype),r.disposeIntermediateTensorInfo(n)}if(null!=o){let e=bU({inputs:{x:d},backend:r,attrs:{perm:tN.getUndoAxesPermutation(o)}});return r.disposeIntermediateTensorInfo(d),r.disposeIntermediateTensorInfo(l),e}return d}let wv={kernelName:x.Cumprod,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,exclusive:i,reverse:o}=n;return wb(g.Prod,a,r,s,i,o)}},ww={kernelName:x.Cumsum,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{axis:s,exclusive:i,reverse:o}=n;return wb(g.Sum,a,r,s,i,o)}},wC={kernelName:x.DenseBincount,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,weights:s}=t,{size:i,binaryOutput:o}=n;if(1===a.shape.length){let e=x9(r.readSync(a.dataId),r.readSync(s.dataId),s.dtype,s.shape,i);return r.makeTensorInfo([i],s.dtype,e)}if(2===a.shape.length){let e=ye(r.bufferSync(a),r.bufferSync(s),i,o);return r.makeTensorInfo(e.shape,s.dtype,e.values)}throw Error(`Error in denseBincount: input must be at most rank 2, but got rank${a.shape.length}.`)}};class wI{constructor(e,t,r){this.variableNames=["x"],this.outputShape=[],this.outputShape=e,this.blockSize=t,this.dataFormat=r,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${t};
      int offset_h = imod(h, ${t});
      int in_w = w / ${t};
      int offset_w = imod(w, ${t});
      int offset_d = (offset_h * ${t} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return"NHWC"===this.dataFormat?"coords[1]":"coords[2]"}getWidthCoordString(){return"NHWC"===this.dataFormat?"coords[2]":"coords[3]"}getDepthCoordString(){return"NHWC"===this.dataFormat?"coords[3]":"coords[1]"}getOutputDepthSize(){return"NHWC"===this.dataFormat?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return"NHWC"===this.dataFormat?"getX(b, in_h, in_w, in_d)":"getX(b, in_d, in_h, in_w)"}}let wk={kernelName:x.DepthToSpace,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{blockSize:s,dataFormat:i}=n,o=a.shape[0],l="NHWC"===i?a.shape[1]:a.shape[2],u="NHWC"===i?a.shape[2]:a.shape[3],h="NHWC"===i?a.shape[3]:a.shape[1],d=l*s,p=u*s,c=h/(s*s),f=new wI("NHWC"===i?[o,d,p,c]:[o,c,d,p],s,i);return r.runWebGLProgram(f,[a],a.dtype)}};var tN=tN;class wS{constructor(e,t=!1,r=null,n=!1,a=!1){this.variableNames=["x","W"],this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=xE(this.outputShape.length);const s=e.filterHeight,i=e.filterWidth,o=e.outChannels/e.inChannels;let l="",u="";r&&(l=n?`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${r}
        }`:a?`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${r}
        }`:`
          float activation(float x) {
            ${r}
          }
        `,u="result = activation(result);"),t&&this.variableNames.push("bias"),n&&this.variableNames.push("preluActivationWeights"),a&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${l}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${o};
        int q = d2 - d1 * ${o};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${s}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${i}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${t?"result += getBiasAtOutCoords();":""}
        ${u}
        setOutput(result);
      }
    `}}class wT{constructor(e,t=!1,r=null,n=!1,a=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=xE(this.outputShape.length);const s=e.outChannels/e.inChannels,i=e.padInfo.left,o=e.strideWidth,l=e.dilationWidth,u=e.filterHeight,h=e.filterWidth;let d=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let e=0;e<h;e++)d+=`
          vec4 xTexelC${2*e};
          int xTexelC${2*e}Ready;
          vec4 xTexelC${2*e+1};
          int xTexelC${2*e+1}Ready;
          vec4 xC${e};`;d+=`
    for (int r = 0; r < ${u}; r++) {
      `;for(let e=0;e<h;e++)d+=`
          xTexelC${2*e} = vec4(0.0);
          xTexelC${2*e}Ready = 0;
          xTexelC${2*e+1} = vec4(0.0);
          xTexelC${2*e+1}Ready = 0;
          xC${e} = vec4(0.0);`;d+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let e=0;e<(h+1)/2;e++){const t=2*e;if(d+=`
          xC = xCCorner + ${t*l};
          `,1===o){if(t<h&&(i%2==1?(d+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }
              `,1===l&&t>0?d+=`
                xC${t} = vec4(xTexelC${t-2}.zw, xTexelC${t}.xy);
                `:d+=`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${t} = vec4(previous.zw, xTexelC${t}.xy);
                  } else {
                    xC${t} = vec4(0.0, 0.0, xTexelC${t}.xy);
                  }
                  `):d+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                xC${t} = xTexelC${t};
                `,t+1<h)){const e=i%2==0?nR.util.nearestLargerEven(l):l;l%2==0&&i%2==1||l%2!=0&&i%2!=1?(d+=`
                  xCOffset = xC + imod(pads[1], 2) + ${e};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                    xTexelC${t+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${t+1}.zw = vec2(0.0);
                    }
                    xTexelC${t+1}Ready = 1;
                  }
                  `,l>1?d+=`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${t+1} = vec4(previous.zw, xTexelC${t+1}.xy);
                    } else {
                     xC${t+1} = vec4(0.0, 0.0, xTexelC${t+1}.xy);
                    }
                    `:d+=`
                    xC${t+1} = vec4(xTexelC${t}.zw, xTexelC${t+1}.xy);
                    `):1===e?d+=`
                    xC${t+1} = xTexelC${t};
                    `:d+=`
                    xCOffset = xC + ${e};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                      xTexelC${t+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${t+1}.zw = vec2(0.0);
                      }
                      xTexelC${t+1}Ready = 1;
                    }

                    xC${t+1} = xTexelC${t+1};
                    `}}else t<h&&(i%2==1?(d+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${t+1}Ready == 0) {
                  xTexelC${t+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${t+1}.zw = vec2(0.0);
                  }
                  xTexelC${t+1}Ready = 1;
                }

                xC${t} = vec4(xTexelC${t}.zw, xTexelC${t+1}.zw);
              `,t+1<h&&(d+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${t+1} = vec4(xTexelC${t+1}.xy, final.xy);
                `)):(d+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${t}Ready == 0) {
                  xTexelC${t} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${t}.zw = vec2(0.0);
                  }
                  xTexelC${t}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${t+1}Ready == 0) {
                  xTexelC${t+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${t+1}.zw = vec2(0.);
                  }
                  xTexelC${t+1}Ready = 1;
                }

                xC${t} = vec4(
                  xTexelC${t}.xy, xTexelC${t+1}.xy);
              `,t+1<h&&(d+=`
                  xC${t+1} = vec4(xTexelC${t}.zw, xTexelC${t+1}.zw);
                `)));t<h&&(d+=`
            wTexel = getW(r, ${t}, d1, q);
            dotProd += xC${t} * vec4(wTexel.xz, wTexel.xz);
          `,t+1<h&&(d+=`
              wTexel = getW(r, ${t+1}, d1, q);
              dotProd += xC${t+1} * vec4(wTexel.xz, wTexel.xz);
            `))}d+=`
    }
  
      }
    `;let p="",c="";r&&(p=n?`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${r}
        }`:a?`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${r}
        }`:`vec4 activation(vec4 x) {
          ${r}
        }`,c="result = activation(result);"),t&&this.variableNames.push("bias"),n&&this.variableNames.push("preluActivationWeights"),a&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${p}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${s};
        int q = d2 - d1 * ${s};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${d}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${t?"result += getBiasAtOutCoords();":""}
        ${c}
        setOutput(result);
      }
    `}}let wN={kernelName:x.DepthwiseConv2dNative,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s,filter:i}=r,{strides:o,pad:l,dilations:u,dimRoundingMode:h}=a,d=u;null==d&&(d=[1,1]),nR.util.assert(tN.eitherStridesOrDilationsAreOne(o,d),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${o} and dilations '${d}'`);let p=tN.computeConv2DInfo(s.shape,i.shape,o,d,l,h,!0);t=(0,nN.env)().getBool("WEBGL_PACK_DEPTHWISECONV")&&p.strideWidth<=2&&p.outChannels/p.inChannels==1?new wT(p):new wS(p);let c=[[p.padInfo.top,p.padInfo.left],[p.strideHeight,p.strideWidth],[p.dilationHeight,p.dilationWidth],[p.inHeight,p.inWidth]];return n.runWebGLProgram(t,[s,i],"float32",c)}};var tN=tN;class w${constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideHeight,r=e.strideWidth,n=e.padInfo.top,a=e.padInfo.left,s=e.outChannels/e.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${s} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${n};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${r} - ${a};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class wR{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterHeight,r=e.filterWidth,n=e.strideHeight,a=e.strideWidth,s=t-1-e.padInfo.top,i=r-1-e.padInfo.left,o=e.outChannels/e.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${s}, ${i});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${n}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${r}; wC++) {
            float dyC = float(dyCCorner + wC) / ${a}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${r} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${o}; dm++) {
              int d2 = d1 * ${o} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}let wA={kernelName:x.DepthwiseConv2dNativeBackpropFilter,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,dy:s}=t,{strides:i,dilations:o,pad:l,dimRoundingMode:u,filterShape:h}=n,d=new w$(tN.computeConv2DInfo(a.shape,h,i,o,l,u,!0));return r.runWebGLProgram(d,[a,s],"float32")}};var tN=tN;let wE={kernelName:x.DepthwiseConv2dNativeBackpropInput,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,filter:s}=t,{strides:i,dilations:o,pad:l,dimRoundingMode:u,inputShape:h}=n,d=new wR(tN.computeConv2DInfo(h,s.shape,i,o,l,u,!0));return r.runWebGLProgram(d,[a,s],"float32")}};class wF{constructor(e){this.variableNames=["X"],this.outputShape=[e,e],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}}let wD={kernelName:x.Diag,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{x:n}=t,a=[...n.shape,...n.shape],s=nR.util.sizeFromShape(n.shape),i=bD({inputs:{x:n},backend:r,attrs:{shape:[s]}}),o=new wF(s),l=r.runWebGLProgram(o,[i],i.dtype),u=bD({inputs:{x:l},backend:r,attrs:{shape:a}});return r.disposeIntermediateTensorInfo(i),r.disposeIntermediateTensorInfo(l),u}};var tN=tN;class wO{constructor(e){this.variableNames=["x","W"],this.outputShape=e.outShape;const{inHeight:t,inWidth:r,padInfo:n,strideHeight:a,strideWidth:s,filterHeight:i,filterWidth:o,dilationHeight:l,dilationWidth:u}=e,{top:h,left:d}=n;this.userCode=`
      const ivec2 strides = ivec2(${a}, ${s});
      const ivec2 pads = ivec2(${h}, ${d});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${i}; h++) {
          int hIn = hBeg + h * ${l};

          if (hIn >= 0 && hIn < ${t}) {
            for (int w = 0; w < ${o}; w++) {
              int wIn = wBeg + w * ${u};

              if (wIn >= 0 && wIn < ${r}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}}let wL={kernelName:x.Dilation2D,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s,filter:i}=r,{strides:o,pad:l,dilations:u}=a,h=tN.computeDilation2DInfo(s.shape,i.shape,o,l,"NHWC",u),d=new wO(h),p=bD({inputs:{x:t=n.runWebGLProgram(d,[s,i],"float32")},backend:n,attrs:{shape:h.outShape}});return n.disposeIntermediateTensorInfo(t),p}};var tN=tN;let wz={kernelName:x.Einsum,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{equation:a}=n,{allDims:s,summedDims:i,idDims:o}=tN.decodeEinsumEquation(a,t.length);tN.checkEinsumDimSizes(s.length,o,t);let{path:l,steps:u}=tN.getEinsumComputePath(i,o),h=u.length,d=null,p=s.length,c=[];for(let e=0;e<h;++e){for(let n of u[e]){let e,{permutationIndices:a,expandDims:s}=tN.getEinsumPermutation(p,o[n]);tN.isIdentityPermutation(a)?e=t[n]:(e=bU({inputs:{x:t[n]},backend:r,attrs:{perm:a}}),c.push(e));let i=e.shape.slice();for(let e=0;e<s.length;++e)i.splice(s[e],0,1);nR.util.arraysEqual(e.shape,i)||(e=bD({inputs:{x:e},backend:r,attrs:{shape:i}}),c.push(e)),null===d?d=e:(d=bE({inputs:{a:e,b:d},backend:r}),c.push(d))}e<h-1&&(l[e]>=0&&(d=bW({inputs:{x:d},backend:r,attrs:{axis:l[e]-(s.length-p),keepDims:!1}}),c.push(d)),p--)}for(let e of c)e!==d&&r.disposeIntermediateTensorInfo(e);return d}},w_=bS({opSnippet:"return (x >= 0.0) ? x : (exp(x) - 1.0);",packedOpSnippet:`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`}),wM={kernelName:x.Elu,backendName:"webgl",kernelFunc:w_},wP=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,wB={kernelName:x.EluGrad,backendName:"webgl",kernelFunc:e=>{let{inputs:t,backend:r}=e,{dy:n,y:a}=t,s=(0,nN.env)().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new bc(wP,n.shape,a.shape):new bd("return (b >= 0.0) ? a : a * (b + 1.0);",n.shape,a.shape);return r.runWebGLProgram(s,[n,a],n.dtype)}},wW=bT({opSnippet:"return float(a == b);",packedOpSnippet:`
  return vec4(equal(a, b));
`,dtype:"bool",cpuKernelImpl:ys}),wG={kernelName:x.Equal,backendName:"webgl",kernelFunc:wW};var tN=tN;let wU=bS({opSnippet:`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${tN.ERF_P};
  float a1 = ${tN.ERF_A1};
  float a2 = ${tN.ERF_A2};
  float a3 = ${tN.ERF_A3};
  float a4 = ${tN.ERF_A4};
  float a5 = ${tN.ERF_A5};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`}),wV={kernelName:x.Erf,backendName:"webgl",kernelFunc:wU},wH=bS({opSnippet:bk+`
  return exp(x);
`,packedOpSnippet:`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:yi,dtype:"float32"}),wq={kernelName:x.Exp,backendName:"webgl",kernelFunc:wH};function wj(e){let{inputs:t,attrs:r,backend:n}=e,{dim:a}=r,{input:s}=t,i=s.shape.length,o=s.shape.slice(),l=a;return a<0&&(nR.util.assert(-(i+1)<=a,()=>`Axis must be in the interval [${-(i+1)}, ${i}]`),l=i+a+1),o.splice(l,0,1),bD({inputs:{x:s},backend:n,attrs:{shape:o}})}let wX={kernelName:x.ExpandDims,backendName:"webgl",kernelFunc:wj},wK="return exp(x) - 1.0;",wY=bS({opSnippet:wK,packedOpSnippet:wK,cpuKernelImpl:yo}),wZ={kernelName:x.Expm1,backendName:"webgl",kernelFunc:wY};class wJ{constructor(e,t,r){let n;this.variableNames=["real","imag"];const a=t[1];this.outputShape=t;const s=r?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,i=r?`${a}.0`:"1.0";if("real"===e)n="return real * expR - imag * expI;";else if("imag"===e)n="return real * expI + imag * expR;";else throw Error(`FFT component must be either "real" or "imag", got ${e}.`);this.userCode=`
      const float exponentMultiplier = ${s};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${n}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${a});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${a}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${i};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}}function wQ(e,t,r){let n=r.texData.get(e.dataId),a=nR.util.sizeFromShape(e.shape),s=e.shape[e.shape.length-1],i=bD({inputs:{x:e},backend:r,attrs:{shape:[a/s,s]}}),o=i.shape,l=new wJ("real",o,t),u=new wJ("imag",o,t),h=[{dataId:n.complexTensorInfos.real.dataId,dtype:n.complexTensorInfos.real.dtype,shape:o},{dataId:n.complexTensorInfos.imag.dataId,dtype:n.complexTensorInfos.imag.dtype,shape:o}],d=r.runWebGLProgram(l,h,"float32"),p=r.runWebGLProgram(u,h,"float32"),c=bg({inputs:{real:d,imag:p},backend:r});r.disposeIntermediateTensorInfo(d),r.disposeIntermediateTensorInfo(p);let f=bD({inputs:{x:c},backend:r,attrs:{shape:e.shape}});return r.disposeIntermediateTensorInfo(i),r.disposeIntermediateTensorInfo(c),f}let w0={kernelName:x.FFT,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{input:n}=t;return wQ(n,!1,r)}};class w1{constructor(e,t){this.outputShape=[],this.customUniforms=[{name:"value",type:"float"}],this.variableNames=["x"],this.outputShape=e,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}}function w2(e){let{backend:t,attrs:r}=e,{shape:n,value:a}=r,{dtype:s}=r;if("string"===(s=s||nR.util.inferDtype(a))){let e=nR.util.getArrayFromDType(s,nR.util.sizeFromShape(n));return e.fill(a),t.makeTensorInfo(n,s,e)}{let e=new w1(n,a),r=[[a]];return t.runWebGLProgram(e,[],s,r)}}let w3={kernelName:x.Fill,backendName:"webgl",kernelFunc:w2};class w4{constructor(e){this.variableNames=["Image"],this.outputShape=[];const t=e[2];this.outputShape=e,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${t} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${t}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}}let w5={kernelName:x.FlipLeftRight,backendName:"webgl",kernelFunc:({inputs:e,backend:t})=>{let{image:r}=e,n=new w4(r.shape);return t.runWebGLProgram(n,[r],r.dtype)}},w6="return floor(x);",w8=bS({opSnippet:w6,packedOpSnippet:w6,cpuKernelImpl:yl}),w7={kernelName:x.Floor,backendName:"webgl",kernelFunc:w8},w9=bT({opSnippet:`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,packedOpSnippet:`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,dtype:"int32"}),Ce={kernelName:x.FloorDiv,backendName:"webgl",kernelFunc:w9};class Ct{constructor(e){this.variableNames=["A"];const t=xd(),[r,n]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${n}.0, ${r}.0);

        vec4 values = ${t.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}}class Cr{constructor(e){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0;const t=xd(),[r,n]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${n}.0, ${r}.0);
            vec4 values = ${t.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${t.output} = result;
      }
    `}}let Cn={kernelName:x.FromPixels,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{pixels:s}=t,{numChannels:i}=n,o="u">typeof HTMLVideoElement&&s instanceof HTMLVideoElement,l="u">typeof HTMLImageElement&&s instanceof HTMLImageElement,[u,h]=o?[s.videoWidth,s.videoHeight]:[s.width,s.height],d=[h,u],p=[h,u,i];if(l||o){let e=(0,nN.env)().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");(null==a||e!==Ca)&&(Ca=e,a=document.createElement("canvas").getContext("2d",{willReadFrequently:Ca})),a.canvas.width=u,a.canvas.height=h,a.drawImage(s,0,0,u,h),s=a.canvas}let c=r.makeTensorInfo(d,"int32");r.texData.get(c.dataId).usage=f.PIXELS,r.gpgpu.uploadPixelDataToTexture(r.getTexture(c.dataId),s);let m=(0,nN.env)().getBool("WEBGL_PACK")?new Cr(p):new Ct(p),g=r.runWebGLProgram(m,[c],"int32");return r.disposeData(c.dataId),g}},Ca=(0,nN.env)().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");var tN=tN;let Cs={kernelName:x.FusedConv2D,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s,filter:i,bias:o,preluActivationWeights:l}=r,{strides:u,pad:h,dataFormat:d,dilations:p,dimRoundingMode:c,activation:f,leakyreluAlpha:m}=a,g=tN.convertConv2DDataFormat(d),x=tN.computeConv2DInfo(s.shape,i.shape,u,p,h,c,!1,g),y=[],b=null!=o,v=null!=l,w="leakyrelu"===f,C=()=>{let e=[s,i],t=(e,t)=>{if("NCHW"===t&&1===e.shape.length&&1!==e.shape[0]){let t=bD({inputs:{x:e},backend:n,attrs:{shape:[e.shape[0],1,1]}});return y.push(t),t}return e};if(b&&e.push(t(o,d)),v&&e.push(t(l,d)),w){let t=n.makeTensorInfo([],"float32",nR.util.createScalarValue(m,"float32"));e.push(t),y.push(t)}return e};if(1===x.filterHeight&&1===x.filterWidth&&1===x.dilationHeight&&1===x.dilationWidth&&1===x.strideHeight&&1===x.strideWidth&&("SAME"===x.padInfo.type||"VALID"===x.padInfo.type))t=v8({x:s,filter:i,convInfo:x,backend:n,bias:o,activation:f,preluActivationWeights:l,leakyreluAlpha:m});else if(x.strideWidth<=2&&"channelsLast"===g&&(0,nN.env)().getBool("WEBGL_EXP_CONV")){let e=new v4(x,b,f?bN(f,!0):null,v,w),r=[[x.padInfo.top,x.padInfo.left],[x.strideHeight,x.strideWidth],[x.dilationHeight,x.dilationWidth],[x.inHeight,x.inWidth]],a=C();t=n.runWebGLProgram(e,a,"float32",r)}else if((0,nN.env)().getBool("WEBGL_CONV_IM2COL"))t=v7({x:s,filter:i,convInfo:x,backend:n,bias:o,activation:f,preluActivationWeights:l,leakyreluAlpha:m});else{let e=new v2(x,b,f?bN(f,!1):null,v,w),r=C();t=n.runWebGLProgram(e,r,"float32")}let I=bD({inputs:{x:t},backend:n,attrs:{shape:x.outShape}});return y.push(t),y.forEach(e=>n.disposeIntermediateTensorInfo(e)),I}};var tN=tN;let Ci={kernelName:x.FusedDepthwiseConv2D,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s,filter:i,bias:o,preluActivationWeights:l}=r,{strides:u,pad:h,dilations:d,dimRoundingMode:p,activation:c,leakyreluAlpha:f}=a,m=[],g=d;null==g&&(g=[1,1]),nR.util.assert(tN.eitherStridesOrDilationsAreOne(u,g),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${u} and dilations '${g}'`);let x=tN.computeConv2DInfo(s.shape,i.shape,u,g,h,p,!0),y=(0,nN.env)().getBool("WEBGL_PACK_DEPTHWISECONV")&&x.strideWidth<=2&&x.outChannels/x.inChannels==1,b=c?bN(c,y):null,v=[s,i],w=null!=o,C=null!=l,I="leakyrelu"===c;if(w&&v.push(o),C&&v.push(l),I){let e=n.makeTensorInfo([],"float32",nR.util.createScalarValue(f,"float32"));v.push(e),m.push(e)}t=y?new wT(x,w,b,C,I):new wS(x,w,b,C,I);let k=[[x.padInfo.top,x.padInfo.left],[x.strideHeight,x.strideWidth],[x.dilationHeight,x.dilationWidth],[x.inHeight,x.inWidth]],S=n.runWebGLProgram(t,v,"float32",k);return m.forEach(e=>n.disposeIntermediateTensorInfo(e)),S}};var tN=tN;class Co{constructor(e,t,r,n){this.sliceDim=e,this.strides=t,this.paramsShape=n,this.variableNames=["x","indices"],this.outputShape=r;const a=xS(r.length);let s=`
    int index;`;for(let e=0;e<this.sliceDim;e++)s+=`
          index = round(getIndices(coords[0], ${e}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[e]};
          flattenIndex += index * ${this.strides[e]};`;this.userCode=`
         void main() {
          ${a} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${s}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}}let Cl={kernelName:x.GatherNd,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{params:n,indices:a}=t,s=a.shape,i=s[s.length-1],o=nR.util.sizeFromShape(n.shape),[l,u,h,d]=tN.prepareAndValidate(n,a),p=bD({inputs:{x:a},backend:r,attrs:{shape:[u,i]}}),c=bD({inputs:{x:n},backend:r,attrs:{shape:[nR.util.sizeFromShape(n.shape)/h,h]}});if(r.shouldExecuteOnCPU([n,a])||"string"===n.dtype){let e=yu(r.readSync(a.dataId),r.bufferSync(n),n.dtype,u,i,h,d,n.shape,o);return r.makeTensorInfo(l,n.dtype,e.values)}let f=new Co(i,d,[u,h],n.shape),m=r.runWebGLProgram(f,[c,p],c.dtype),g=bD({inputs:{x:m},backend:r,attrs:{shape:l}});return r.disposeIntermediateTensorInfo(p),r.disposeIntermediateTensorInfo(c),r.disposeIntermediateTensorInfo(m),g}};var tN=tN;class Cu{constructor(e,t){this.variableNames=["A","indices"],this.outputShape=t,this.rank=t.length;const r=xS(this.rank),n=function(e){let t=["resRC.x","resRC.y","resRC.z","resRC.w"],r=[];for(let n=0;n<e.length;n++)2===n?r.push("index"):r.push(`${t[n]}`);return r.join()}(e);this.userCode=`
      void main() {
        ${r} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${e[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${n}));
      }
    `}}function Ch(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,indices:s}=t,{axis:i,batchDims:o}=n,l=nR.util.parseAxisParam(i,a.shape)[0];if((0,nN.env)().get("DEBUG")){let e=r.readSync(s.dataId),t=a.shape[l];for(let r=0;r<e.length;++r){let n=e[r];nR.util.assert(n<=t-1&&n>=0,()=>`GatherV2: the index value ${n} is not in [0, ${t-1}]`)}}let u=tN.segment_util.collectGatherOpShapeInfo(a,s,l,o),h=nR.util.sizeFromShape(s.shape),d=[],p=bD({inputs:{x:a},backend:r,attrs:{shape:[u.batchSize,u.outerSize,u.dimSize,u.sliceSize]}}),c=bD({inputs:{x:s},backend:r,attrs:{shape:[u.batchSize,h/u.batchSize]}});d.push(p),d.push(c);let f=[u.batchSize,u.outerSize,h/u.batchSize,u.sliceSize];if(r.shouldExecuteOnCPU([a,s])||"string"===a.dtype){let e=r.bufferSync(c),t=yh(r.bufferSync(p),e,f);return d.forEach(e=>r.disposeIntermediateTensorInfo(e)),r.makeTensorInfo(u.outputShape,t.dtype,t.values)}let m=new Cu(p.shape,f),g=r.runWebGLProgram(m,[p,c],p.dtype);d.push(g);let x=bD({inputs:{x:g},backend:r,attrs:{shape:u.outputShape}});return d.forEach(e=>r.disposeIntermediateTensorInfo(e)),x}let Cd={kernelName:x.GatherV2,backendName:"webgl",kernelFunc:Ch},Cp=bT({opSnippet:"return float(a > b);",packedOpSnippet:`
  return vec4(greaterThan(a, b));
`,cpuKernelImpl:yd,dtype:"bool"}),Cc={kernelName:x.Greater,backendName:"webgl",kernelFunc:Cp},Cf=bT({opSnippet:"return float(a >= b);",packedOpSnippet:`
  return vec4(greaterThanEqual(a, b));
`,dtype:"bool",cpuKernelImpl:yp}),Cm={kernelName:x.GreaterEqual,backendName:"webgl",kernelFunc:Cf},Cg={kernelName:x.IFFT,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{input:n}=t;return wQ(n,!0,r)}},Cx=bS({opSnippet:"return float(!isnan(x) && !isinf(x));",dtype:"bool"}),Cy={kernelName:x.IsFinite,backendName:"webgl",kernelFunc:Cx},Cb=bS({opSnippet:"return float(isinf(x));",dtype:"bool"}),Cv={kernelName:x.IsInf,backendName:"webgl",kernelFunc:Cb},Cw=bS({opSnippet:"return float(isnan(x));",dtype:"bool"}),CC={kernelName:x.IsNan,backendName:"webgl",kernelFunc:Cw},CI=bT({opSnippet:"return float(a < b);",packedOpSnippet:`
  return vec4(lessThan(a, b));
`,cpuKernelImpl:yc,dtype:"bool"}),Ck={kernelName:x.Less,backendName:"webgl",kernelFunc:CI},CS=bT({opSnippet:"return float(a <= b);",packedOpSnippet:`
  return vec4(lessThanEqual(a, b));
`,cpuKernelImpl:yf,dtype:"bool"}),CT={kernelName:x.LessEqual,backendName:"webgl",kernelFunc:CS},CN={kernelName:x.LinSpace,backendName:"webgl",kernelFunc:function(e){let{backend:t,attrs:r}=e,{start:n,stop:a,num:s}=r,i=ym(n,a,s);return t.makeTensorInfo([i.length],"float32",i)}},C$=bS({opSnippet:bk+`
  return x < 0.0 ? 0./0. : log(x);
`,packedOpSnippet:`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,cpuKernelImpl:yg}),CR={kernelName:x.Log,backendName:"webgl",kernelFunc:C$},CA=bS({opSnippet:bk+`
  return log(1.0 + x);
`}),CE={kernelName:x.Log1p,backendName:"webgl",kernelFunc:CA},CF=bT({opSnippet:"return float(a >= 1.0 && b >= 1.0);",packedOpSnippet:`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,dtype:"bool"}),CD={kernelName:x.LogicalAnd,backendName:"webgl",kernelFunc:CF},CO=bS({opSnippet:"return float(!(x >= 1.0));"}),CL={kernelName:x.LogicalNot,backendName:"webgl",kernelFunc:CO},Cz=bT({opSnippet:"return float(a >= 1.0 || b >= 1.0);",packedOpSnippet:`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,dtype:"bool"}),C_={kernelName:x.LogicalOr,backendName:"webgl",kernelFunc:Cz};class CM{constructor(e,t,r,n,a){let s;this.variableNames=["x"],this.outputShape=[];const i=e[3]-1;this.outputShape=e;const o=`float(${r}) + float(${n}) * sum`;s=.5===a?`inversesqrt(${o})`:1===a?`1.0/(${o})`:`exp(log(${o}) * float(-${a}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${t}; j <= ${t}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${i}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${s};
        setOutput(val);
      }
    `}}class CP{constructor(e,t,r,n,a){let s;this.variableNames=["x"],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;const i=e[3]-1;this.outputShape=e;const o=`float(${r}) + float(${n}) * sum`;s=.5===a?`inversesqrt(${o})`:1===a?`1.0/(${o})`:`exp(log(${o}) * float(-${a}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${t};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${t}; j <= ${t}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${i}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${s};
        setOutput(result);
      }
    `}}let CB={kernelName:x.LRN,backendName:"webgl",kernelFunc:e=>{let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{depthRadius:s,bias:i,alpha:o,beta:l}=n,u=(0,nN.env)().getBool("WEBGL_PACK_NORMALIZATION")?new CP(a.shape,s,i,o,l):new CM(a.shape,s,i,o,l);return r.runWebGLProgram(u,[a],a.dtype)}};class CW{constructor(e,t,r,n,a){this.variableNames=["inputImage","outputImage","dy"],this.outputShape=[],this.outputShape=e,this.depth=e[3],this.depthRadius=t,this.bias=r,this.alpha=n,this.beta=a,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${t})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${t} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${n}) * norm + float(${r});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${n})
                * float(${a})
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${a});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}}let CG={kernelName:x.LRNGrad,backendName:"webgl",kernelFunc:e=>{let{inputs:t,backend:r,attrs:n}=e,{x:a,y:s,dy:i}=t,{depthRadius:o,bias:l,alpha:u,beta:h}=n,d=new CW(a.shape,o,l,u,h);return r.runWebGLProgram(d,[a,s,i],a.dtype)}};var tN=tN;function CU(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{reductionIndices:i,keepDims:o}=a,l=s.shape.length,u=nR.util.parseAxisParam(i,s.shape),h=u,d=tN.getAxesPermutation(h,l),p=null!=d,c=n.shouldExecuteOnCPU([s]),f=s;if(p){if(c){let e=n.texData.get(f.dataId).values,t=Array(l);for(let e=0;e<t.length;e++)t[e]=s.shape[d[e]];let r=yH(e,s.shape,s.dtype,d,t);f=n.makeTensorInfo(t,s.dtype),n.texData.get(f.dataId).values=r}else f=bB(s,d,n);h=tN.getInnerMostAxes(h.length,l)}tN.assertAxesAreInnerMostDims("max",h,l);let[m,g]=tN.computeOutAndReduceShapes(f.shape,h),x=m;if(o&&(x=tN.expandShapeToKeepDim(m,u)),c){let e=yx(n.texData.get(f.dataId).values,nR.util.sizeFromShape(g),x,s.dtype);t=n.makeTensorInfo(x,s.dtype),n.texData.get(t.dataId).values=e}else{var y,b;let e,r,a,s,i;y=f,b=x,e=nR.util.sizeFromShape(g),r=nR.util.sizeFromShape(y.shape),s=b_(a=bD({inputs:{x:y},attrs:{shape:[r/e,e]},backend:n}),y.dtype,"max",n),i=bD({inputs:{x:s},attrs:{shape:b},backend:n}),n.disposeIntermediateTensorInfo(a),n.disposeIntermediateTensorInfo(s),t=i}return p&&n.disposeIntermediateTensorInfo(f),t}let CV={kernelName:x.Max,backendName:"webgl",kernelFunc:CU},CH=bT({opSnippet:bh+`
  return max(a, b);
`,packedOpSnippet:`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+bp+`
  return result;
`,cpuKernelImpl:yy}),Cq={kernelName:x.Maximum,backendName:"webgl",kernelFunc:CH};var tN=tN;let Cj={kernelName:x.MaxPool,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t;xl(a,"maxPool");let{filterSize:s,strides:i,pad:o,dimRoundingMode:l}=n;nR.util.assert(tN.eitherStridesOrDilationsAreOne(i,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${i} and dilations '1'`);let u=tN.computePool2DInfo(a.shape,s,i,1,o,l);if(1===u.filterWidth&&1===u.filterHeight&&nR.util.arraysEqual(u.inShape,u.outShape))return bf({inputs:{x:a},backend:r});let h=new vp(u,"max",!1);return r.runWebGLProgram(h,[a],a.dtype)}};var tN=tN;let CX={kernelName:x.MaxPool3D,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{filterSize:s,strides:i,pad:o,dataFormat:l,dimRoundingMode:u}=n,h=new vc(tN.computePool3DInfo(a.shape,s,i,[1,1,1],o,u,l),"max",!1);return r.runWebGLProgram(h,[a],a.dtype)}};var tN=tN;class CK{constructor(e){this.variableNames=["dy","maxPos"],this.outputShape=e.inShape;const t=e.strideHeight,r=e.strideWidth,n=e.dilationHeight,a=e.effectiveFilterHeight,s=e.effectiveFilterWidth,i=a-1-e.padInfo.top,o=s-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${i}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${a};
          wR += ${n}) {
          float dyR = float(dyRCorner + wR) / ${t}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${a*s-1} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${s} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}}class CY{constructor(e){this.variableNames=["dy","maxPos"],this.outputShape=e.inShape;const t=e.strideDepth,r=e.strideHeight,n=e.strideWidth,a=e.dilationDepth,s=e.dilationHeight,i=e.dilationWidth,o=e.effectiveFilterDepth,l=e.effectiveFilterHeight,u=e.effectiveFilterWidth,h=o-1-e.padInfo.front,d=l-1-e.padInfo.top,p=u-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${h}, ${d}, ${p});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${o};
           wD += ${a}) {
          float dyD = float(dyDCorner + wD) / ${t}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${l};
              wR += ${s}) {
            float dyR = float(dyRCorner + wR) / ${r}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${u};
                wC += ${i}) {
              float dyC = float(dyCCorner + wC) / ${n}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${o*l*u-1} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${l} * ${u} +
                  wR * ${u} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}let CZ={kernelName:x.MaxPool3DGrad,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,input:s}=t,{filterSize:i,strides:o,pad:l,dimRoundingMode:u}=n,h=tN.computePool3DInfo(s.shape,i,o,[1,1,1],l,u),d=new vc(h,"max",!0),p=r.runWebGLProgram(d,[s],s.dtype),c=new CY(h),f=r.runWebGLProgram(c,[a,p],s.dtype);return r.disposeIntermediateTensorInfo(p),f}};var tN=tN;let CJ={kernelName:x.MaxPoolGrad,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{dy:a,input:s,output:i}=t;xl([s,i],"maxPoolGrad");let{filterSize:o,strides:l,pad:u,dimRoundingMode:h}=n,d=tN.computePool2DInfo(s.shape,o,l,1,u,h),p=new vp(d,"max",!0),c=r.runWebGLProgram(p,[s],s.dtype),f=new CK(d),m=r.runWebGLProgram(f,[a,c],s.dtype);return r.disposeIntermediateTensorInfo(c),m}};var tN=tN;let CQ={kernelName:x.MaxPoolWithArgmax,backendName:"webgl",kernelFunc:({inputs:e,attrs:t,backend:r})=>{let n,a,{x:s}=e,{filterSize:i,strides:o,pad:l,includeBatchInIndex:u}=t;nR.util.assert(4===s.shape.length,()=>`Error in maxPool: input must be rank 4 but got rank ${s.shape.length}.`);let h=[1,1];nR.util.assert(tN.eitherStridesOrDilationsAreOne(o,h),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '${h}'`);let d=tN.computePool2DInfo(s.shape,i,o,h,l),[p,c]=(n=new vp(d,"max",!1),a=r.runWebGLProgram(n,[s],"float32"),n=new vp(d,"max",!0,!0,u),[a,r.runWebGLProgram(n,[s],"float32")]);return[p,c]}};var tN=tN;let C0={kernelName:x.Mean,backendName:"webgl",kernelFunc:({inputs:e,attrs:t,backend:r})=>{var n,a;let s,i,o,l,u,{x:h}=e,{keepDims:d,axis:p}=t,c=h.shape.length,f=nR.util.parseAxisParam(p,h.shape),m=f,g=tN.getAxesPermutation(m,c),x=null!=g,y=r.shouldExecuteOnCPU([h]),b=[],v=h;if(x){if(y){let e=r.texData.get(v.dataId).values,t=Array(c);for(let e=0;e<t.length;e++)t[e]=h.shape[g[e]];let n=yH(e,h.shape,h.dtype,g,t);v=r.makeTensorInfo(t,h.dtype),r.texData.get(v.dataId).values=n}else v=bB(h,g,r);b.push(v),m=tN.getInnerMostAxes(m.length,c)}tN.assertAxesAreInnerMostDims("sum",m,c);let[w,C]=tN.computeOutAndReduceShapes(v.shape,m),I=w;d&&(I=tN.expandShapeToKeepDim(w,f));let k=(n=v,a=I,s=nR.util.sizeFromShape(C),i=nR.util.sizeFromShape(n.shape),l=b_(o=bD({inputs:{x:n},attrs:{shape:[i/s,s]},backend:r}),"float32","mean",r),u=bD({inputs:{x:l},attrs:{shape:a},backend:r}),r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(l),u);for(let e of b)r.disposeIntermediateTensorInfo(e);return k}};var tN=tN;let C1={kernelName:x.Min,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{axis:i,keepDims:o}=a,l=s.shape.length,u=nR.util.parseAxisParam(i,s.shape),h=u,d=tN.getAxesPermutation(h,l),p=s;null!=d&&(p=bU({inputs:{x:s},backend:n,attrs:{perm:d}}),h=tN.getInnerMostAxes(h.length,s.shape.length)),tN.assertAxesAreInnerMostDims("min",h,l);let[c,f]=tN.computeOutAndReduceShapes(p.shape,h),m=bD({inputs:{x:p},backend:n,attrs:{shape:[-1,nR.util.sizeFromShape(f)]}}),g=b_(m,m.dtype,"min",n);return t=o?bD({inputs:{x:g},backend:n,attrs:{shape:tN.expandShapeToKeepDim(c,u)}}):bD({inputs:{x:g},backend:n,attrs:{shape:c}}),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(g),null!=d&&n.disposeIntermediateTensorInfo(p),t}},C2=bT({opSnippet:bh+`
  return min(a, b);
`,packedOpSnippet:`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+bp+`
  return result;
`,cpuKernelImpl:yb}),C3={kernelName:x.Minimum,backendName:"webgl",kernelFunc:C2};class C4{constructor(e,t,r){this.variableNames=["x"],this.outputShape=t.map((t,r)=>t[0]+e[r]+t[1]);const n=e.length,a=xS(n),s=t.map(e=>e[0]).join(","),i=t.map((t,r)=>t[0]+e[r]).join(","),o=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,n),l=+("reflect"!==r);if(1===n){this.userCode=`
        int start = ${s};
        int end = ${i};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${l};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${l};
          }
          setOutput(getX(outC - start));
        }
      `;return}this.userCode=`
      ${a} start = ${a}(${s});
      ${a} end = ${a}(${i});

      void main() {
        ${a} outC = getOutputCoords();
        for (int i = 0; i < ${n}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${l};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${l};
          }
        }
        ${a} coords = outC - start;
        setOutput(getX(${o}));
      }
    `}}class C5{constructor(e,t,r){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t.map((t,r)=>t[0]+e[r]+t[1]);const n=e.length,a=xS(n),s=t.map(e=>e[0]).join(","),i=t.map((t,r)=>t[0]+e[r]).join(","),o=yX("rc",n),l=yX("source",n),u=`${o[n-1]} < ${this.outputShape[n-1]}`,h=1===n?"source":`vec2(${l.slice(-2).join()})`,d=+("reflect"!==r);let p="";if(1===n){const e=`
        ${a} source = rc;
        if (source < start) {
          source = start * 2 - source - ${d};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${d};
        }
        source -= start;
      `;p=`
        ${a} rc = outputLoc;
        ${e}
        result[0] = getChannel(getX(${l.join()}), ${h});
        ${o[n-1]} += 1;
        if(${u}) {
          ${e}
          result[1] = getChannel(getX(${l.join()}), ${h});
        }
      `}else{const e=`
        ${a} source = rc;
        ${a} lt = ${a}(lessThan(source, start));
        ${a} gte = ${a}(greaterThanEqual(source, end));
        ${a} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${d}) +
                gte * ((end - 1) * 2 - source + ${d});
        source -= start;
      `;p=`
        ${a} rc = outputLoc;
        ${e}
        result[0] = getChannel(getX(${l.join()}), ${h});
        ${o[n-1]} += 1;
        if(${u}) {
          ${e}
          result[1] = getChannel(getX(${l.join()}), ${h});
        }
        rc = outputLoc;
        ${o[n-2]} += 1;
        if(${o[n-2]} < ${this.outputShape[n-2]}) {
          ${e}
          result[2] = getChannel(getX(${l.join()}), ${h});
          ${o[n-1]} += 1;
          if(${u}) {
            ${e}
            result[3] = getChannel(getX(${l.join()}), ${h});
          }
        }
      `}this.userCode=`
      const ${a} start = ${a}(${s});
      const ${a} end = ${a}(${i});

      void main() {
        ${a} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${p}
        setOutput(result);
      }
    `}}let C6={kernelName:x.MirrorPad,backendName:"webgl",kernelFunc:({inputs:e,backend:t,attrs:r})=>{let{x:n}=e,{paddings:a,mode:s}=r,i=(0,nN.env)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new C5(n.shape,a,s):new C4(n.shape,a,s);return t.runWebGLProgram(i,[n],n.dtype)}},C8=bT({opSnippet:`if (b == 0.0) return NAN;
  return mod(a, b);`,packedOpSnippet:`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+bp+`
  return result;
`}),C7={kernelName:x.Mod,backendName:"webgl",kernelFunc:C8};class C9{constructor(e,t,r){this.variableNames=["probs"],this.customUniforms=[{name:"seed",type:"float"}],this.outputShape=[e,r],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${t-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${t-1}));
      }
    `}}var tN=tN;let Ie=bT({opSnippet:`
if (a == b) {
  return 1.0;
};
return a / b;`,packedOpSnippet:`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,checkOutOfBounds:!0}),It={kernelName:x.RealDiv,backendName:"webgl",kernelFunc:Ie},Ir="return a - b;",In=bT({opSnippet:Ir,packedOpSnippet:Ir,supportsComplex:!0,cpuKernelImpl:yG}),Ia={kernelName:x.Sub,backendName:"webgl",kernelFunc:In};function Is(e){let{inputs:t,backend:r,attrs:n}=e,{logits:a}=t,{dim:s}=n,i=nR.util.parseAxisParam([s],a.shape),o=CU({inputs:{x:a},backend:r,attrs:{reductionIndices:i,keepDims:!1}}),l=tN.expandShapeToKeepDim(o.shape,i),u=bD({inputs:{x:o},backend:r,attrs:{shape:l}}),h=In({inputs:{a:a,b:u},backend:r}),d=wH({inputs:{x:h},backend:r}),p=bW({inputs:{x:d},backend:r,attrs:{axis:i,keepDims:!1}}),c=bD({inputs:{x:p},backend:r,attrs:{shape:l}}),f=Ie({inputs:{a:d,b:c},backend:r});return r.disposeIntermediateTensorInfo(o),r.disposeIntermediateTensorInfo(u),r.disposeIntermediateTensorInfo(h),r.disposeIntermediateTensorInfo(d),r.disposeIntermediateTensorInfo(p),r.disposeIntermediateTensorInfo(c),f}let Ii={kernelName:x.Softmax,backendName:"webgl",kernelFunc:Is},Io={kernelName:x.Multinomial,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{logits:a}=t,{numSamples:s,seed:i,normalized:o}=n,l=o?a:Is({inputs:{logits:a},backend:r,attrs:{dim:a.shape.length-1}}),u=new C9(l.shape[0],l.shape[1],s),h=r.runWebGLProgram(u,[l],"int32",[[i]]);return o||r.disposeIntermediateTensorInfo(l),h}},Il=y2+`
  return -x;
`,Iu=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Ih={kernelName:x.Neg,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n}=e,{x:a}=r;if(n.shouldExecuteOnCPU([a])){let[e,t]=yw(n.texData.get(a.dataId).values,a.shape,a.dtype);return n.makeTensorInfo(t,a.dtype,e)}return t=(0,nN.env)().getBool("WEBGL_PACK_UNARY_OPERATIONS")?new be(a.shape,Iu):new y1(a.shape,Il),n.runWebGLProgram(t,[a],a.dtype)}};var tN=tN,h8=h8;let Id=h8.nonMaxSuppressionV3Impl,Ip={kernelName:x.NonMaxSuppressionV3,backendName:"webgl",kernelFunc:function(e){tN.warn("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");let{inputs:t,backend:r,attrs:n}=e,{boxes:a,scores:s}=t,{maxOutputSize:i,iouThreshold:o,scoreThreshold:l}=n,{selectedIndices:u}=Id(r.readSync(a.dataId),r.readSync(s.dataId),i,o,l);return r.makeTensorInfo([u.length],"int32",new Int32Array(u))}};var tN=tN,h8=h8;let Ic=h8.nonMaxSuppressionV4Impl,If={kernelName:x.NonMaxSuppressionV4,backendName:"webgl",kernelFunc:function(e){tN.warn("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");let{inputs:t,backend:r,attrs:n}=e,{boxes:a,scores:s}=t,{maxOutputSize:i,iouThreshold:o,scoreThreshold:l,padToMaxOutputSize:u}=n,{selectedIndices:h,validOutputs:d}=Ic(r.readSync(a.dataId),r.readSync(s.dataId),i,o,l,u);return[r.makeTensorInfo([h.length],"int32",new Int32Array(h)),r.makeTensorInfo([],"int32",new Int32Array([d]))]}};var tN=tN,h8=h8;let Im=h8.nonMaxSuppressionV5Impl,Ig={kernelName:x.NonMaxSuppressionV5,backendName:"webgl",kernelFunc:function(e){tN.warn("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");let{inputs:t,backend:r,attrs:n}=e,{boxes:a,scores:s}=t,{maxOutputSize:i,iouThreshold:o,scoreThreshold:l,softNmsSigma:u}=n,{selectedIndices:h,selectedScores:d}=Im(r.readSync(a.dataId),r.readSync(s.dataId),i,o,l,u);return[r.makeTensorInfo([h.length],"int32",new Int32Array(h)),r.makeTensorInfo([d.length],"float32",new Float32Array(d))]}};class Ix{constructor(e,t,r,n){this.variableNames=["indices"],this.outputShape=[e,t],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${n}), float(${r}),
                      float(index == coords.y)));
      }
    `}}let Iy={kernelName:x.OneHot,backendName:"webgl",kernelFunc:e=>{let{inputs:t,backend:r,attrs:n}=e,{indices:a}=t,{dtype:s,depth:i,onValue:o,offValue:l}=n,u=nR.util.sizeFromShape(a.shape),h=new Ix(u,i,o,l),d=bD({inputs:{x:a},backend:r,attrs:{shape:[u]}}),p=r.runWebGLProgram(h,[d],s);r.disposeIntermediateTensorInfo(d);let c=bD({inputs:{x:p},backend:r,attrs:{shape:[...a.shape,i]}});return r.disposeIntermediateTensorInfo(p),c}};function Ib(e){let{inputs:t,backend:r}=e,{x:n}=t;if("complex64"!==n.dtype)return w2({attrs:{shape:n.shape,dtype:n.dtype,value:"string"===n.dtype?"":0},backend:r});{let e=v_({inputs:{input:n},backend:r}),t=Ib({inputs:{x:e},backend:r}),a=vJ({inputs:{input:n},backend:r}),s=Ib({inputs:{x:a},backend:r}),i=bg({inputs:{real:t,imag:s},backend:r});return r.disposeIntermediateTensorInfo(e),r.disposeIntermediateTensorInfo(t),r.disposeIntermediateTensorInfo(a),r.disposeIntermediateTensorInfo(s),i}}let Iv={kernelName:x.ZerosLike,backendName:"webgl",kernelFunc:Ib},Iw={kernelName:x.OnesLike,backendName:"webgl",kernelFunc:function e(t){let{inputs:r,backend:n}=t,{x:a}=r;if("string"===a.dtype)throw Error("onesLike is not supported under string dtype");if("complex64"!==a.dtype)return w2({attrs:{shape:a.shape,dtype:a.dtype,value:1},backend:n});{let t=v_({inputs:{input:a},backend:n}),r=e({inputs:{x:t},backend:n}),s=vJ({inputs:{input:a},backend:n}),i=Ib({inputs:{x:s},backend:n}),o=bg({inputs:{real:r,imag:i},backend:n});return n.disposeIntermediateTensorInfo(t),n.disposeIntermediateTensorInfo(r),n.disposeIntermediateTensorInfo(s),n.disposeIntermediateTensorInfo(i),o}}},IC={kernelName:x.Pack,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{axis:a}=n;if(1===t.length)return wj({inputs:{input:t[0]},backend:r,attrs:{dim:a}});let s=t[0].shape,i=t[0].dtype;t.forEach(e=>{nR.util.assertShapesMatch(s,e.shape,"All tensors passed to stack must have matching shapes"),nR.util.assert(i===e.dtype,()=>"All tensors passed to stack must have matching dtypes")});let o=[],l=v0({inputs:t.map(e=>{let t=wj({inputs:{input:e},backend:r,attrs:{dim:a}});return o.push(t),t}),backend:r,attrs:{axis:a}});return o.forEach(e=>r.disposeIntermediateTensorInfo(e)),l}};class II{constructor(e,t,r){this.variableNames=["x"],this.customUniforms=[{name:"value",type:"float"}],this.outputShape=t.map((t,r)=>t[0]+e[r]+t[1]);const n=e.length,a=xS(n),s=t.map(e=>e[0]).join(","),i=t.map((t,r)=>t[0]+e[r]).join(","),o=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,n);if(1===n){this.userCode=`
        int start = ${s};
        int end = ${i};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;return}this.userCode=`
      ${a} start = ${a}(${s});
      ${a} end = ${a}(${i});

      void main() {
        ${a} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${a} coords = outC - start;
          setOutput(getX(${o}));
        }
      }
    `}}class Ik{constructor(e,t,r){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"value",type:"float"}],this.outputShape=t.map((t,r)=>t[0]+e[r]+t[1]);const n=e.length,a=xS(n),s=t.map(e=>e[0]).join(","),i=t.map((t,r)=>t[0]+e[r]).join(","),o=yX("rc",n),l=yX("source",n),u=`${o[n-1]} < ${this.outputShape[n-1]}`,h=1===n?"source":`vec2(${l.slice(-2).join()})`,d=[`${a} rc = outputLoc;`,`${o[n-1]} += 1;
       if(${u}) {
      `,1===n?"":`}
       rc = outputLoc;
       ${o[n-2]} += 1;
       if(${o[n-2]} < ${this.outputShape[n-2]}) {`,1===n?"":`  ${o[n-1]} += 1;
         if(${u}) {`],p=1===n?"rc < start || rc >= end":"any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))";let c="";for(let e=0,t=1===n?2:4;e<t;e++)c+=`
        ${d[e]}
        if (${p}) {
          result[${e}] = float(value);
        } else {
          ${a} source = rc - start;
          result[${e}] = getChannel(getX(${l.join()}), ${h});
        }
      `;c+=1===n?"} ":"}}",this.userCode=`
      const ${a} start = ${a}(${s});
      const ${a} end = ${a}(${i});

      void main() {
        ${a} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${c}
        setOutput(result);
      }
    `}}let IS=e=>{let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{paddings:s,constantValue:i}=n;if(0===nR.util.sizeFromShape(a.shape))return w2({backend:r,attrs:{shape:s.map((e,t)=>e[0]+a.shape[t]+e[1]),value:i,dtype:a.dtype}});let o=(0,nN.env)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new Ik(a.shape,s,i):new II(a.shape,s,i),l=[[i]];return r.runWebGLProgram(o,[a],a.dtype,l)},IT={kernelName:x.PadV2,backendName:"webgl",kernelFunc:IS},IN=bT({opSnippet:`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,packedOpSnippet:`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+bp+`
  return result;
`}),I$={kernelName:x.Pow,backendName:"webgl",kernelFunc:IN};var tN=tN;let IR={kernelName:x.Prod,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{axis:i,keepDims:o}=a,l=s.shape.length,u=[],h=nR.util.parseAxisParam(i,s.shape),d=h,p=tN.getAxesPermutation(d,l),c=s;if(null!=p&&(c=bU({inputs:{x:s},backend:n,attrs:{perm:p}}),d=tN.getInnerMostAxes(d.length,l),u.push(c)),tN.assertAxesAreInnerMostDims("prod",d,l),n.shouldExecuteOnCPU([c])){let e=n.texData.get(c.dataId).values,{outVals:r,outShape:a,outDtype:s}=yI(c.shape,c.dtype,e,d);t=n.makeTensorInfo(a,s,r)}else{let[e,r]=tN.computeOutAndReduceShapes(c.shape,d),a=bD({inputs:{x:c},backend:n,attrs:{shape:[-1,nR.util.sizeFromShape(r)]}}),i=b_(a,(0,py.sumOutType)(s.dtype),"prod",n);t=bD({inputs:{x:i},backend:n,attrs:{shape:e}}),u.push(a),u.push(i)}if(o){u.push(t);let e=tN.expandShapeToKeepDim(t.shape,h);t=bD({inputs:{x:t},backend:n,attrs:{shape:e}})}return u.forEach(e=>n.disposeIntermediateTensorInfo(e)),t}},IA={kernelName:x.RaggedGather,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{paramsNestedSplits:a,paramsDenseValues:s,indices:i}=t,{outputRaggedRank:o}=n,l=a.map(e=>r.readSync(e.dataId)),u=a.map(e=>e.shape),h=r.readSync(s.dataId),d=r.readSync(i.dataId),[p,c,f]=yk(l,u,h,s.shape,s.dtype,d,i.shape,o),m=p.map(e=>r.makeTensorInfo([e.length],"int32",e)),g=r.makeTensorInfo(f,s.dtype,c);return m.concat([g])}},IE={kernelName:x.RaggedRange,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{starts:n,limits:a,deltas:s}=t,i=r.readSync(n.dataId),o=r.readSync(a.dataId),l=r.readSync(s.dataId),[u,h]=yS(i,n.shape,n.dtype,o,a.shape,l,s.shape);return[r.makeTensorInfo([u.length],"int32",u),r.makeTensorInfo([h.length],n.dtype,h)]}},IF={kernelName:x.RaggedTensorToTensor,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{shape:a,values:s,defaultValue:i,rowPartitionTensors:o}=t,{rowPartitionTypes:l}=n,u=r.readSync(a.dataId),h=r.readSync(s.dataId),d=r.readSync(i.dataId),p=o.map(e=>r.readSync(e.dataId)),c=o.map(e=>e.shape),[f,m]=yT(u,a.shape,h,s.shape,s.dtype,d,i.shape,p,c,l);return r.makeTensorInfo(f,s.dtype,m)}},ID=e=>{let{backend:t,attrs:r}=e,{start:n,stop:a,step:s,dtype:i}=r,o=yN(n,a,s,i);return t.makeTensorInfo([o.length],i,o)},IO={kernelName:x.Range,backendName:"webgl",kernelFunc:ID},IL=bS({opSnippet:"return 1.0 / x;"}),Iz={kernelName:x.Reciprocal,backendName:"webgl",kernelFunc:IL},I_=bS({opSnippet:y2+`
  return (x < 0.0) ? 0.0 : x;
`,packedOpSnippet:`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`}),IM={kernelName:x.Relu,backendName:"webgl",kernelFunc:I_},IP=bS({opSnippet:y2+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,packedOpSnippet:`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`}),IB={kernelName:x.Relu6,backendName:"webgl",kernelFunc:IP};class IW{constructor(e,t,r,n,a){this.variableNames=["A"],this.outputShape=[];const[s,i,o,l]=e;this.outputShape=[s,t,r,l];const u=[n&&t>1?i-1:i,n&&r>1?o-1:o],h=[n&&t>1?t-1:t,n&&r>1?r-1:r];this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/h[0]},
          ${u[1]/h[1]});
      const vec2 inputShapeRC = vec2(${i}.0, ${o}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${a?"(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)":"vec2(yRC) * effectiveInputOverOutputRatioRC"};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}}class IG{constructor(e,t,r,n,a){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[s,i,o,l]=e;this.outputShape=[s,t,r,l];const u=[n&&t>1?i-1:i,n&&r>1?o-1:o],h=[n&&t>1?t-1:t,n&&r>1?r-1:r];this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/h[0]},
          ${u[1]/h[1]},
          ${u[1]/h[1]});
      const vec3 inputShapeRC = vec3(${i}.0, ${o}.0,
                                     ${o}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${a?"(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)":"vec3(yRC) * effectiveInputOverOutputRatioRC"};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${l-1};
        bool hasNextRow = coords.z < ${r-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}}let IU={kernelName:x.ResizeBilinear,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{images:a}=t,{alignCorners:s,halfPixelCenters:i,size:o}=n,[l,u]=o,h=(0,nN.env)().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new IG(a.shape,l,u,s,i):new IW(a.shape,l,u,s,i);return r.runWebGLProgram(h,[a],"float32")}};class IV{constructor(e,t,r){this.variableNames=["dy"],this.outputShape=[],this.outputShape=t;const[,n,a]=t,[,s,i]=e,o=[r&&s>1?n-1:n,r&&i>1?a-1:a],l=[r&&s>1?s-1:s,r&&i>1?i-1:i],u=o[0]/l[0],h=o[1]/l[1],d=1/u,p=1/h,c=2*Math.ceil(d)+2,f=2*Math.ceil(p)+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${h});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${p});

        const int winHeight = int(${c});
        const int winWidth = int(${f});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${s}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${i}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${n-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${a-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}let IH={kernelName:x.ResizeBilinearGrad,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{images:a,dy:s}=t,{alignCorners:i}=n,o=new IV(s.shape,a.shape,i);return r.runWebGLProgram(o,[s],s.dtype)}};class Iq{constructor(e,t,r,n,a){this.variableNames=["A"],this.outputShape=[];const[s,i,o,l]=e;this.outputShape=[s,t,r,l];const u=[n&&t>1?i-1:i,n&&r>1?o-1:o],h=[n&&t>1?t-1:t,n&&r>1?r-1:r];this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/h[0]},
          ${u[1]/h[1]});
      const vec2 inputShapeRC = vec2(${i}.0, ${o}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${a?"max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))":"vec2(yRC) * effectiveInputOverOutputRatioRC"};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${n?"0.5":"0.0"})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}}class Ij{constructor(e,t,r,n,a){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[s,i,o,l]=e;this.outputShape=[s,t,r,l];const u=[n&&t>1?i-1:i,n&&r>1?o-1:o],h=[n&&t>1?t-1:t,n&&r>1?r-1:r];this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/h[0]},
          ${u[1]/h[1]},
          ${u[1]/h[1]});
      const vec3 inputShapeRC = vec3(${i}.0, ${o}.0,
                                     ${o}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${a?"max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))":"vec3(yRC) * effectiveInputOverOutputRatioRC"};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${n?"0.5":"0.0"})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${l-1};
        bool hasNextRow = coords.z < ${r-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}}let IX={kernelName:x.ResizeNearestNeighbor,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{images:a}=t,{alignCorners:s,halfPixelCenters:i,size:o}=n,[l,u]=o,h=(0,nN.env)().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new Ij(a.shape,l,u,s,i):new Iq(a.shape,l,u,s,i);return r.runWebGLProgram(h,[a],a.dtype)}};class IK{constructor(e,t,r){this.variableNames=["dy"],this.outputShape=[],this.outputShape=t;const[,n,a]=t,[,s,i]=e,o=[r&&s>1?n-1:n,r&&i>1?a-1:a],l=[r&&s>1?s-1:s,r&&i>1?i-1:i],u=o[0]/l[0],h=o[1]/l[1],d=1/u,p=1/h,c=2*Math.ceil(d)+2,f=2*Math.ceil(p)+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${h});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${p});

        const int winHeight = int(${c});
        const int winWidth = int(${f});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${s}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${i}) {
              continue;
            }

            float sourceFracRow =
              float(${o[0]}) *
                (float(dyR) / float(${l[0]}));

            float sourceFracCol =
                float(${o[1]}) *
                  (float(dyC) / float(${l[1]}));

            int sourceNearestRow = int(min(
                float(int(${n}) - 1),
                ${r} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${a}) - 1),
                ${r} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}let IY={kernelName:x.ResizeNearestNeighborGrad,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{images:a,dy:s}=t,{alignCorners:i}=n,o=new IK(s.shape,a.shape,i);return r.runWebGLProgram(o,[s],s.dtype)}};class IZ{constructor(e,t){this.variableNames=["x"];const r=e.length;if(r>4)throw Error(`WebGL backend: Reverse of rank-${r} tensor is not yet supported`);if(this.outputShape=e,1===r){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${e[0]} - coord - 1));
        }
      `;return}const n=e.map((r,n)=>-1!==t.indexOf(n)&&1!==e[n]?`${e[n]} - coords[${n}] - 1`:`coords[${n}]`).join(","),a=xS(r);this.userCode=`
      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${n}));
      }
    `}}class IJ{constructor(e,t){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0;const r=e.length;if(r>4)throw Error(`WebGL backend: Reverse of rank-${r} tensor is not yet supported`);this.outputShape=e;const n=yX("rc",r),a=`${n[r-1]} + 1 < ${this.outputShape[r-1]}`,s=`${n[r-2]} + 1 < ${this.outputShape[r-2]}`,i=xS(r);function o(r){let n=e.map((n,a)=>{var s,i;return s=a,i=r,-1!==t.indexOf(s)&&1!==e[s]?`${e[s]} - ${i[s]} - 1`:`${i[s]}`}),a=n.join(","),s=n.slice(-2).join(",");return`getChannel(getX(${a}), vec2(${s}))`}1===r?this.userCode=`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${e[0]} - rc - 1),
            ${e[0]} - rc - 1);
          if(${a}){
              result.g = getChannel(getX(${e[0]} - (rc  + 1) - 1),
                ${e[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:this.userCode=`
        void main() {
          ${i} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${function(e){return o(e)}(n.slice())};
          if(${a}){
            result.g = ${function(e){return e[r-1]="("+e[r-1]+" + 1)",o(e)}(n.slice())};
          }
          if(${s}) {
            result.b = ${function(e){return e[r-2]="("+e[r-2]+" + 1)",o(e)}(n.slice())};
            if(${a}) {
              result.a = ${function(e){return e[r-1]="("+e[r-1]+" + 1)",e[r-2]="("+e[r-2]+" + 1)",o(e)}(n.slice())};
            }
          }
          setOutput(result);
        }
    `}}let IQ={kernelName:x.Reverse,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{dims:s}=n,i=a.shape.length,o=nR.util.parseAxisParam(s,a.shape);if(0===i)return bf({inputs:{x:a},backend:r});let l=(0,nN.env)().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new IJ(a.shape,o):new IZ(a.shape,o);return r.runWebGLProgram(l,[a],a.dtype)}};var tN=tN;class I0{constructor(e,t){this.variableNames=["Image"],this.outputShape=[],this.customUniforms=[{name:"params",type:"vec4"}];const r=e[1],n=e[2];this.outputShape=e;let a="";a="number"==typeof t?`float outputValue = ${t.toFixed(2)};`:`
        vec3 fill = vec3(${t.join(",")});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${a}
          if(coordX >= 0 && coordX < ${n} && coordY >= 0 && coordY < ${r}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}}let I1={kernelName:x.RotateWithOffset,backendName:"webgl",kernelFunc:({inputs:e,attrs:t,backend:r})=>{let{image:n}=e,{radians:a,fillValue:s,center:i}=t,o=new I0(n.shape,s),[l,u]=tN.getImageCenter(i,n.shape[1],n.shape[2]),h=[[l,u,Math.sin(a),Math.cos(a)]];return r.runWebGLProgram(o,[n],n.dtype,h)}},I2=bS({opSnippet:`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`}),I3={kernelName:x.Round,backendName:"webgl",kernelFunc:I2},I4=bS({opSnippet:"return inversesqrt(x);",cpuKernelImpl:y$}),I5={kernelName:x.Rsqrt,backendName:"webgl",kernelFunc:I4};var tN=tN;class I6{constructor(e,t,r,n,a,s,i=!0,o=!1){this.variableNames=["updates","indices","defaultValue"],this.outputShape=s;const l=xS(a.length),u=xS(s.length);let h="";1===r?h="i":2===r&&(h="i, j");const d=`getIndices(${h})`;let p="";1===n?p="i":2===n&&(p="i, coords[1]");const c=`getUpdates(${p})`;let f="";o&&(f="coords[0], coords[1]");const m=`getDefaultValue(${f})`;this.userCode=`
        ${l} strides = ${l}(${a});

        void main() {
          ${u} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${e}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${t}; j++) {
              int index = round(${d});
              flattenedIndex += index * ${t>1?"strides[j]":"strides"};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${c};
              found = true;
            }
          }
          setOutput(mix(${m}, sum, float(found)));
        }
      `}}class I8{constructor(e,t,r,n,a,s,i=!0,o=!1){this.variableNames=["updates","indices","defaultValue"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=s;const l=xS(a.length),u=xS(s.length);let h="";1===r?h="i":2===r&&(h="i, j");const d=`getIndices(${h})`;let p="";1===n?p="i":2===n&&(p="i, coords[1]");const c=`getUpdates(${p})`;let f="";o&&(f="coords[0], coords[1]");const m=`getDefaultValue(${f})`;this.userCode=`
        ${l} strides = ${l}(${a});

        void main() {
          ${u} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${e}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${t}; j+=2) {
              ivec4 index = round(${d});
              flattenedIndex += index.xz * ${t>1?"strides[j]":"strides"};
              if (j + 1 < ${t}) {
                flattenedIndex += index.yw * ${t>1?"strides[j + 1]":"strides"};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${c};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${m}, sum, found));
        }
      `}}let I7={kernelName:x.ScatterNd,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{indices:s,updates:i}=r,{shape:o}=a,{sliceRank:l,numUpdates:u,sliceSize:h,strides:d,outputSize:p}=tN.calculateShapes(i,s,o),c=[p/h,h];if(0===p)return n.makeTensorInfo(o,s.dtype);let f=bD({inputs:{x:s},backend:n,attrs:{shape:[u,l]}}),m=bD({inputs:{x:i},backend:n,attrs:{shape:[u,h]}}),g=n.makeTensorInfo([],"float32",new Float32Array([0]));t=(0,nN.env)().getBool("WEBGL_PACK")?new I8(u,l,f.shape.length,m.shape.length,d,c):new I6(u,l,f.shape.length,m.shape.length,d,c);let x=n.runWebGLProgram(t,[m,f,g],m.dtype),y=bD({inputs:{x:x},backend:n,attrs:{shape:o}});return n.disposeIntermediateTensorInfo(f),n.disposeIntermediateTensorInfo(m),n.disposeIntermediateTensorInfo(x),n.disposeIntermediateTensorInfo(g),y}};class I9{constructor(e,t,r,n){this.variableNames=["sortedSequence","values"],this.customUniforms=[{name:"numInputs",type:"int"}],this.outputShape=[e,r];const a=`for (int i = 0; i < ${Math.ceil(Math.log2(t+1))}; ++i) { if (left >= right) break;`,s=2===(0,nN.env)().getNumber("WEBGL_VERSION")?"while (left < right) {":a;this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${s}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${"left"===n?"<":"<="} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}}let ke={kernelName:x.SearchSorted,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{sortedSequence:a,values:s}=t,{side:i}=n,o=new I9(a.shape[0],a.shape[1],s.shape[1],i),l=[[a.shape[1]]];return r.runWebGLProgram(o,[a,s],"int32",l)}};class kt{constructor(e,t,r){let n,a;if(this.variableNames=["c","a","b"],this.outputShape=t,r>4)throw Error(`Where for rank ${r} is not yet supported`);if(1===r)a="resRC",n="resRC";else{const r=["resRC.x","resRC.y","resRC.z","resRC.w"],s=[],i=[];for(let n=0;n<t.length;n++)i.push(`${r[n]}`),n<e&&s.push(`${r[n]}`);n=s.join(),a=i.join()}const s=xS(r);this.userCode=`
      void main() {
        ${s} resRC = getOutputCoords();
        float cVal = getC(${n});
        if (cVal >= 1.0) {
          setOutput(getA(${a}));
        } else {
          setOutput(getB(${a}));
        }
      }
    `}}let kr={kernelName:x.Select,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{condition:n,t:a,e:s}=t,i=new kt(n.shape.length,a.shape,a.shape.length);return r.runWebGLProgram(i,[n,a,s],(0,py.upcastType)(a.dtype,s.dtype))}};var tN=tN;let kn=bS({opSnippet:`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${tN.SELU_SCALEALPHA};
  float scale = ${tN.SELU_SCALE};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`}),ka={kernelName:x.Selu,backendName:"webgl",kernelFunc:kn},ks=bS({opSnippet:bk+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,packedOpSnippet:`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:yA}),ki={kernelName:x.Sigmoid,backendName:"webgl",kernelFunc:ks},ko=bS({opSnippet:`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`}),kl={kernelName:x.Sign,backendName:"webgl",kernelFunc:ko},ku=bS({opSnippet:bk+`
  return sin(x);
`,packedOpSnippet:`
  vec4 result = sin(x);
  bvec4 isNaN = isnan(x);
  ${bp}
  return result;
`}),kh={kernelName:x.Sin,backendName:"webgl",kernelFunc:ku},kd=bS({opSnippet:`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`}),kp={kernelName:x.Sinh,backendName:"webgl",kernelFunc:kd},kc=bS({opSnippet:`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`}),kf={kernelName:x.Softplus,backendName:"webgl",kernelFunc:kc};var tN=tN;let km={kernelName:x.SpaceToBatchND,backendName:"webgl",kernelFunc:e=>{let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{blockShape:s,paddings:i}=n;nR.util.assert(a.shape.length<=4,()=>"spaceToBatchND for rank > 4 with a WebGL backend not implemented yet");let o=s.reduce((e,t)=>e*t),l=[[0,0]];l.push(...i);for(let e=1+s.length;e<a.shape.length;++e)l.push([0,0]);let u=[],h=IS({inputs:{x:a},backend:r,attrs:{paddings:l,constantValue:0}}),d=tN.getReshaped(h.shape,s,o,!1),p=tN.getPermuted(d.length,s.length,!1),c=tN.getReshapedPermuted(h.shape,s,o,!1),f=bD({inputs:{x:h},backend:r,attrs:{shape:d}}),m=bU({inputs:{x:f},backend:r,attrs:{perm:p}}),g=bD({inputs:{x:m},backend:r,attrs:{shape:c}});return u.push(h),u.push(f),u.push(m),u.forEach(e=>r.disposeIntermediateTensorInfo(e)),g}},kg={kernelName:x.SparseFillEmptyRows,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{indices:n,values:a,denseShape:s,defaultValue:i}=t;if(1!==s.shape.length)throw Error(`Dense shape must be a vector, saw:
         ${s.shape}`);if(2!==n.shape.length)throw Error(`Indices must be a matrix, saw:
         ${n.shape}`);if(1!==a.shape.length)throw Error(`Values must be a vector, saw:
         ${a.shape}`);if(0!==i.shape.length)throw Error(`Default value must be a scalar, saw:
        ${i.shape}`);let o=r.readSync(n.dataId),l=r.readSync(a.dataId),u=r.readSync(s.dataId),h=r.readSync(i.dataId)[0],[d,p,c,f,m]=yD(o,n.shape,n.dtype,l,a.dtype,u,h);return[r.makeTensorInfo(p,n.dtype,d),r.makeTensorInfo([p[0]],a.dtype,c),r.makeTensorInfo([f.length],"bool",new Uint8Array(f.map(e=>Number(e)))),r.makeTensorInfo([m.length],n.dtype,new Int32Array(m))]}},kx={kernelName:x.SparseReshape,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{inputIndices:n,inputShape:a,newShape:s}=t;if(2!==n.shape.length)throw Error(`Input indices should be a matrix but received shape ${n.shape}`);if(1!==a.shape.length)throw Error(`Input shape should be a vector but received shape ${a.shape}`);if(1!==s.shape.length)throw Error(`Target shape should be a vector but received shape ${s.shape}`);let i=Array.from(r.readSync(a.dataId)),o=r.readSync(n.dataId),l=Array.from(r.readSync(s.dataId)),[u,h,d]=yO(o,n.shape,n.dtype,i,l);return[r.makeTensorInfo(h,n.dtype,u),r.makeTensorInfo([d.length],s.dtype,new Int32Array(d))]}},ky={kernelName:x.SparseSegmentMean,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{data:n,indices:a,segmentIds:s}=t;if(n.shape.length<1)throw Error("Data should be at least 1 dimensional but received scalar");if(1!==a.shape.length)throw Error(`Indices should be a vector but received shape
              ${a.shape}`);if(1!==s.shape.length)throw Error(`Segment ids should be a vector but received shape
              ${s.shape}`);let i=r.readSync(n.dataId),o=r.readSync(a.dataId),l=r.readSync(s.dataId),[u,h]=yL(i,n.shape,n.dtype,o,l,!0);return r.makeTensorInfo(h,n.dtype,u)}},kb={kernelName:x.SparseSegmentSum,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r}=e,{data:n,indices:a,segmentIds:s}=t;if(n.shape.length<1)throw Error("Data should be at least 1 dimensional but received scalar");if(1!==a.shape.length)throw Error(`Indices should be a vector but received shape
             ${a.shape}`);if(1!==s.shape.length)throw Error(`Segment ids should be a vector but received shape
             ${s.shape}`);let i=r.readSync(n.dataId),o=r.readSync(a.dataId),l=r.readSync(s.dataId),[u,h]=yL(i,n.shape,n.dtype,o,l);return r.makeTensorInfo(h,n.dtype,u)}};var tN=tN;let kv={kernelName:x.SparseToDense,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{sparseIndices:a,sparseValues:s,defaultValue:i}=t,{outputShape:o}=n,{sliceRank:l,numUpdates:u,sliceSize:h,strides:d,outputSize:p}=tN.calculateShapes(s,a,o);if("string"===s.dtype){let e=yR(r.bufferSync(a),r.bufferSync(s),o,p,h,u,l,d,nR.util.decodeString(r.readSync(i.dataId)[0]),!1);return r.makeTensorInfo(o,e.dtype,e.values)}let c=new I6(u,l,a.shape.length,s.shape.length,d,[p,1],!1),f=r.runWebGLProgram(c,[s,a,i],s.dtype),m=bD({inputs:{x:f},backend:r,attrs:{shape:o}});return r.disposeIntermediateTensorInfo(f),m}};var tN=tN;let kw={kernelName:x.SplitV,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{numOrSizeSplits:s,axis:i}=n,o=nR.util.parseAxisParam(i,a.shape)[0],l=tN.prepareSplitSize(a,s,o),u=Array(a.shape.length).fill(0),h=a.shape.slice();return l.map(e=>{let t=[...h];t[o]=e;let n=vN({inputs:{x:a},backend:r,attrs:{begin:u,size:t}});return u[o]+=e,n})}},kC="return sqrt(x);",kI=bS({opSnippet:kC,packedOpSnippet:kC,cpuKernelImpl:yz}),kk={kernelName:x.Sqrt,backendName:"webgl",kernelFunc:kI},kS=bS({opSnippet:"return x * x;"}),kT={kernelName:x.Square,backendName:"webgl",kernelFunc:kS},kN="return (a - b) * (a - b);",k$=bT({opSnippet:kN,packedOpSnippet:kN}),kR={kernelName:x.SquaredDifference,backendName:"webgl",kernelFunc:k$};var tN=tN;let kA={kernelName:x.StaticRegexReplace,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t;if("string"!==a.dtype)throw Error("Input must be of datatype string");let s=r.readSync(a.dataId),i=y_(tN.fromUint8ToStringArray(s),"string",n);return r.makeTensorInfo(a.shape,"string",i)}},kE={kernelName:x.Step,backendName:"webgl",kernelFunc:function({inputs:e,attrs:t,backend:r}){let{x:n}=e,a=y2+`
    return x > 0.0 ? 1.0 : float(${t.alpha});
  `,s=new y1(n.shape,a);return r.runWebGLProgram(s,[n],n.dtype)}};var pB=tJ;class kF{constructor(e,t,r){this.variableNames=["x"],this.outputShape=r;const n=r.length,a=xS(r.length),s=xS(r.length);let i="";if(1===n)i="coords * strides + begin";else{let e=0;i=r.map((t,n)=>(e++,1===r.length?`coords * strides[${n}] + begin[${n}]`:`coords[${e-1}] * strides[${n}] + begin[${n}]`)).join(",")}this.userCode=`
      ${a} begin = ${a}(${e});
      ${a} strides = ${a}(${t});

      void main() {
        ${s} coords = getOutputCoords();
        setOutput(getX(${i}));
      }
    `}}let kD={kernelName:x.StridedSlice,backendName:"webgl",kernelFunc:function(e){let t,{inputs:r,backend:n,attrs:a}=e,{x:s}=r,{begin:i,end:o,strides:l,beginMask:u,endMask:h,ellipsisMask:d,newAxisMask:p,shrinkAxisMask:c}=a,{finalShapeSparse:f,finalShape:m,isIdentity:g,sliceDim0:x,isSimpleSlice:y,begin:b,end:v,strides:w}=pB.sliceInfo(s.shape,i,o,l,u,h,d,p,c);if(g)t=bD({inputs:{x:s},backend:n,attrs:{shape:m}});else if(x||y){nR.util.assert(s.shape.length>=1,()=>`Input must have rank at least 1, got: ${s.shape.length}`);let e=pB.computeOutShape(b,v,w),r=vN({inputs:{x:s},backend:n,attrs:{begin:b,size:e}});t=bD({inputs:{x:r},backend:n,attrs:{shape:m}}),n.disposeIntermediateTensorInfo(r)}else if(n.shouldExecuteOnCPU([s])){let e=n.readSync(s.dataId),r=yM(f,(0,h5.buffer)(s.shape,s.dtype,e),w,b);t=n.makeTensorInfo(m,s.dtype,r.values)}else{let e=new kF(b,w,f);t=n.runWebGLProgram(e,[s],s.dtype)}let C=bD({inputs:{x:t},backend:n,attrs:{shape:m}});return n.disposeIntermediateTensorInfo(t),C}},kO={kernelName:x.StringNGrams,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{separator:a,nGramWidths:s,leftPad:i,rightPad:o,padWidth:l,preserveShortSequences:u}=n,{data:h,dataSplits:d}=t,[p,c]=yP(r.readSync(h.dataId),r.readSync(d.dataId),a,s,i,o,l,u);return[r.makeTensorInfo([p.length],"string",p),r.makeTensorInfo(d.shape,"int32",c)]}},kL={kernelName:x.StringSplit,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{skipEmpty:a}=n,{input:s,delimiter:i}=t;if("string"!==s.dtype)throw Error("Input must be of datatype string");if(1!==s.shape.length)throw Error(`Input must be a vector, got shape: ${s.shape}`);if(0!==i.shape.length)throw Error(`Delimiter must be a scalar, got shape: ${i.shape}`);let[o,l,u]=yB(r.readSync(s.dataId),r.readSync(i.dataId)[0],a),h=l.length;return[r.makeTensorInfo([h,2],"int32",o),r.makeTensorInfo([h],"string",l),r.makeTensorInfo([2],"int32",new Int32Array(u))]}},kz={kernelName:x.StringToHashBucketFast,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{numBuckets:a}=n,{input:s}=t;if("string"!==s.dtype)throw Error("Input must be of datatype string");if(a<=0)throw Error("Number of buckets must be at least 1");let i=yW(r.readSync(s.dataId),a);return r.makeTensorInfo(s.shape,"int32",i)}},k_=bS({opSnippet:"return tan(x);"}),kM={kernelName:x.Tan,backendName:"webgl",kernelFunc:k_},kP=bS({opSnippet:`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`}),kB={kernelName:x.Tanh,backendName:"webgl",kernelFunc:kP};var tN=tN;let kW={kernelName:x.TensorScatterUpdate,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{tensor:a,indices:s,updates:i}=t,{}=n,{sliceRank:o,numUpdates:l,sliceSize:u,strides:h,outputSize:d}=tN.calculateShapes(i,s,a.shape),p=[d/u,u];if(0===d)return r.makeTensorInfo(a.shape,s.dtype);let c=bD({inputs:{x:s},backend:r,attrs:{shape:[l,o]}}),f=bD({inputs:{x:i},backend:r,attrs:{shape:[l,u]}}),m=bD({inputs:{x:a},backend:r,attrs:{shape:p}}),g=new I6(l,o,c.shape.length,f.shape.length,h,p,!1,!0),x=r.runWebGLProgram(g,[f,c,m],m.dtype),y=bD({inputs:{x:x},backend:r,attrs:{shape:a.shape}});return r.disposeIntermediateTensorInfo(c),r.disposeIntermediateTensorInfo(f),r.disposeIntermediateTensorInfo(m),r.disposeIntermediateTensorInfo(x),y}};class kG{constructor(e,t){this.variableNames=["A"];const r=Array(e.length);for(let n=0;n<r.length;n++)r[n]=e[n]*t[n];this.outputShape=r,this.rank=r.length;const n=xS(this.rank),a=function(e){let t=e.length;if(t>5)throw Error(`Tile for rank ${t} is not yet supported`);if(1===t)return`imod(resRC, ${e[0]})`;let r=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u"],n=[];for(let t=0;t<e.length;t++)n.push(`imod(${r[t]}, ${e[t]})`);return n.join()}(e);this.userCode=`
      void main() {
        ${n} resRC = getOutputCoords();
        setOutput(getA(${a}));
      }
    `}}function kU(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{reps:s}=n;if("string"===a.dtype||a.shape.length>5){let e=r.readSync(a.dataId),t="string"===a.dtype?e.map(e=>nR.util.decodeString(e)):e,n=yU((0,h5.buffer)(a.shape,a.dtype,t),s);return r.makeTensorInfo(n.shape,n.dtype,n.values)}let i=new kG(a.shape,s);return r.runWebGLProgram(i,[a],a.dtype)}let kV={kernelName:x.Tile,backendName:"webgl",kernelFunc:kU};class kH{constructor(e){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"negativeInf",type:"float"},{name:"dir",type:"int"},{name:"inc",type:"int"}],this.outputShape=e,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}}class kq{constructor(e){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"k",type:"int"}],this.outputShape=e,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}}function kj(e,t){null!==t&&e.disposeIntermediateTensorInfo(t)}function kX(e){let t=1;for(;t<e;)t*=2;return t}let kK={kernelName:x.TopK,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a}=t,{k:s,sorted:i}=n,o=(0,nN.env)().getNumber("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD"),l=(0,nN.env)().getNumber("TOPK_K_CPU_HANDOFF_THRESHOLD"),u=a.shape,h=u[u.length-1];if(r.shouldExecuteOnCPU([a])||h<o||s>l){let[e,t]=yV(r.readSync(a.dataId),u,a.dtype,s,i);return[r.makeTensorInfo(e.shape,e.dtype,e.values),r.makeTensorInfo(t.shape,t.dtype,t.values)]}if(0===s)return u[u.length-1]=0,[r.makeTensorInfo(u,a.dtype,[]),r.makeTensorInfo(u,"int32",[])];if(1===h)return[a,w2({attrs:{shape:u,dtype:"int32",value:0},backend:r})];let d=r.texData.get(a.dataId),p=null!==d&&d.isPacked,c=p?r.unpackTensor(a):a,f=nR.util.sizeFromShape(u)/h,m=bD({inputs:{x:c},attrs:{shape:[f,h]},backend:r});p&&kj(r,c);let g=kX(s),x=kX(h),y=null,b=()=>null===y?[m,m]:[m,y],v=(e,t,n)=>{let a=b(),s=new kH(n),i=[[h],[+(null===y)],[-1/0],[e],[t]],o=y;y=r.runWebGLProgram(s,a,"int32",i),kj(r,o)};for(let e=1;e<g;e*=2){let t=2*e;for(let r=e;r>=1;r/=2)v(t,r,[f,x])}for(let e=x;e>g;e/=2){let t=b(),n=new kq([f,e/2]),a=[[h],[+(null===y)],[g]],s=y;y=r.runWebGLProgram(n,t,"int32",a),kj(r,s);let i=g/2,o=2*i;for(let e=i;e>=1;e/=2)v(o,e,y.shape)}let w=y;y=vN({inputs:{x:y},backend:r,attrs:{begin:0,size:[f,s]}}),kj(r,w);let C=Ch({inputs:{x:m,indices:y},backend:r,attrs:{axis:1,batchDims:1}});kj(r,m);let I=u.slice(0,-1);I.push(s),w=y,y=bD({inputs:{x:y},attrs:{shape:I},backend:r}),kj(r,w);let k=C;return C=bD({inputs:{x:C},attrs:{shape:I},backend:r}),kj(r,k),[C,y]}};class kY{constructor(e,t,r,n,a,s){let i;switch(this.variableNames=["Image","Transforms"],this.outputShape=s,n){case"constant":default:i=1;break;case"reflect":i=2;break;case"wrap":i=3;break;case"nearest":i=4}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${i} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${i} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${i} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${e} && 0 <= coordX && coordX < ${t}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${a});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${a});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${t}));
                float mapY = mapCoord(inY, float(${e}));

                if (${"nearest"===r?1:2} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}}let kZ={kernelName:x.Transform,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{image:a,transforms:s}=t,{interpolation:i,fillMode:o,fillValue:l,outputShape:u}=n,[h,d,p,c]=a.shape,[f,m]=null!=u?u:[d,p],g=new kY(d,p,i,o,l,[h,f,m,c]);return r.runWebGLProgram(g,[a,s],"float32")}},kJ={kernelName:x.Unique,backendName:"webgl",kernelFunc:function(e){let{inputs:t,attrs:r,backend:n}=e,{axis:a}=r,{x:s}=t;xl(s,"unique"),console.warn("WARNING: ","UI might be locked temporarily as data is being downloaded");let{outputValues:i,outputShape:o,indices:l}=yq(n.readSync(s.dataId),a,s.shape,s.dtype);return[n.makeTensorInfo(o,s.dtype,i),n.makeTensorInfo([l.length],"int32",l)]}},kQ={kernelName:x.Unpack,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{value:a}=t,{axis:s}=n;s<0&&(s+=a.shape.length);let i=a.shape.length,o=a.shape[s],l=Array(i-1),u=0;for(let e=0;e<i;e++)e!==s&&(l[u++]=a.shape[e]);let h=[],d=Array(i).fill(0),p=a.shape.slice();p[s]=1;let c=Array(o);for(let e=0;e<c.length;e++){d[s]=e;let t=vN({inputs:{x:a},backend:r,attrs:{begin:d,size:p}}),n=bD({inputs:{x:t},backend:r,attrs:{shape:l}});c[e]=n,h.push(t)}return h.forEach(e=>r.disposeIntermediateTensorInfo(e)),c}};var tN=tN;class k0{constructor(e,t){this.variableNames=["x","segmentIds"];const r=e.windowSize,n=e.batchSize,a=e.inSize,s=e.numSegments,i=s*Math.ceil(a/r);this.outputShape=[n,i];const o=4*Math.floor(r/4),l=r%4,u=`
        sumValue += dot(values, segFilter);
    `;let h="";a%r>0&&(h=`
        if (inIdx < 0 || inIdx >= ${a}) {
          return initializationValue;
        }
      `);let d="";a%r>0&&(d=`
        if (inIdx < 0 || inIdx >= ${a}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = 0.0;

      float getValue(int batch, int inIdx) {
        ${h}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${d}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${s})) * float(${r}));
        int currentSeg = int(mod(float(outIdx), float(${s})));

        float sumValue = 0.0;

        for (int i = 0; i < ${o}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${u}
        }

        int inIdx = inOffset + ${o};
        if (${1===l}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${u}
        } else if (${2===l}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${u}
        } else if (${3===l}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${u}
        }
        setOutput(sumValue);
      }
    `}}for(let e of[bq,bX,bY,bJ,b1,b4,b5,b6,ve,vt,vn,vs,vo,vu,vd,vf,vm,vy,vb,vv,vI,vR,vA,vD,vO,vP,vG,vH,bx,vX,v1,v9,wa,wi,wo,wl,wu,wd,wc,wm,wv,ww,wC,wk,wN,wA,wE,wD,wL,wz,wM,wB,wG,wV,wq,wX,wZ,w0,w3,w5,w7,Ce,Cn,Cs,Ci,Cl,Cd,Cc,Cm,bm,Cg,vQ,Cy,Cv,CC,bv,Ck,CT,CN,CR,CE,CD,CL,C_,CB,CG,CV,Cq,Cj,CX,CZ,CJ,CQ,C0,C1,C3,C6,C7,Io,bF,Ih,Ip,If,Ig,vz,Iy,Iw,IC,IT,I$,bI,IR,IA,IE,IF,IO,vM,It,Iz,IM,IB,bO,IU,IH,IX,IY,IQ,I1,I3,I5,I7,ke,kr,ka,ki,kl,kh,kp,v$,Ii,kf,km,kg,kx,ky,kb,kv,kw,kk,kT,kR,kA,kE,kD,kO,kL,kz,Ia,bG,kM,kB,kW,kV,kK,kZ,bV,kJ,kQ,{kernelName:x.UnsortedSegmentSum,backendName:"webgl",kernelFunc:function(e){let{inputs:t,backend:r,attrs:n}=e,{x:a,segmentIds:s}=t,{numSegments:i}=n,o=a.shape.length,l=[],u=0,h=tN.getAxesPermutation([u],o),d=a;null!=h&&(d=bU({inputs:{x:a},backend:r,attrs:{perm:h}}),l.push(d),u=tN.getInnerMostAxes(1,o)[0]);let p=tN.segment_util.computeOutShape(d.shape,u,i),c=nR.util.sizeFromShape([d.shape[u]]),f=bD({inputs:{x:d},backend:r,attrs:{shape:[-1,c]}});l.push(f);let m=(0,py.sumOutType)(a.dtype),g=(e,t,n,a,s)=>{let i=e.shape[0],o=e.shape[1],u=tN.segment_util.segOpComputeOptimalWindowSize(o,s),h=new k0({windowSize:u,inSize:o,batchSize:i,numSegments:s},t),d=r.compileAndRun(h,[e,n],a);if(l.push(d),d.shape[1]===s)return d;let p=ID({backend:r,attrs:{start:0,stop:s,step:1,dtype:"float32"}}),c=kU({inputs:{x:p},backend:r,attrs:{reps:[o/u]}});return l.push(p),l.push(c),g(d,t,c,a,s)},x=bD({inputs:{x:g(f,"unsortedSegmentSum",s,m,i)},backend:r,attrs:{shape:p}}),y=x;return null!=h&&(l.push(x),y=bU({inputs:{x:y},backend:r,attrs:{perm:tN.getUndoAxesPermutation(h)}})),l.forEach(e=>r.disposeIntermediateTensorInfo(e)),y}},Iv])(0,rc.registerKernel)(e);e.s([],22659);var k1=e.i(38589),k1=k1;let k2="4.22.0",k3="4.22.0",k4="4.22.0",k5={"tfjs-core":k1.version,"tfjs-backend-cpu":k2,"tfjs-backend-webgl":k3,"tfjs-data":h3,"tfjs-layers":id,"tfjs-converter":k4,tfjs:"4.22.0"};e.s(["version",0,k5],93112),e.i(93112);var k6=e.i(35943);e.i(48635),e.s(["Callback",0,u6,"CallbackList",0,sB,"CustomCallback",0,sU,"EarlyStopping",0,u9,"History",0,sG,"InputSpec",0,sp,"LayerVariable",0,su,"LayersModel",0,iF,"RNN",0,oF,"Sequential",0,i_,"SymbolicTensor",0,sc,"callbacks",0,he,"constraints",0,sO,"initializers",0,sL,"input",0,iB,"layers",0,u2,"loadLayersModel",0,iL,"metrics",0,u3,"model",0,iM,"models",0,u4,"registerCallbackConstructor",0,iW,"regularizers",0,u5,"sequential",0,iP,"version_layers",0,id],5350),e.i(5350);var k8=e.i(31860),k7=e.i(38657);e.s(["GraphModel",()=>k8.GraphModel,"deregisterOp",()=>k7.deregisterOp,"loadGraphModel",()=>k8.loadGraphModel,"loadGraphModelSync",()=>k8.loadGraphModelSync,"registerOp",()=>k7.registerOp,"version_converter",0,k4],87892),e.i(87892),e.i(18538),e.i(60131),e.s(["MathBackendCPU",0,de,"shared",0,cs,"version_cpu",0,k2],72930),e.i(72930),e.s(["MathBackendCPU",0,de,"shared",()=>cs,"version_cpu",0,k2],55113),e.i(55113),e.i(22659),e.i(51868),e.i(69754),e.s(["GPGPUContext",0,x8,"MathBackendWebGL",0,bs,"forceHalfFloat",0,bl,"gpgpu_util",0,bi,"setWebGLContext",0,gT,"webgl_util",0,bo],86451),e.i(86451),e.s(["GPGPUContext",0,x8,"MathBackendWebGL",0,bs,"forceHalfFloat",0,bl,"gpgpu_util",()=>bi,"setWebGLContext",0,gT,"version_webgl",0,k3,"webgl",0,bu,"webgl_util",()=>bo],37145),e.i(37145),e.s(["GPGPUContext",0,x8,"MathBackendWebGL",0,bs,"forceHalfFloat",0,bl,"gpgpu_util",()=>bi,"setWebGLContext",0,gT,"version_webgl",0,k3,"webgl",0,bu,"webgl_util",()=>bo],81866),e.i(81866),e.s(["Abs",()=>k6.Abs,"Acos",()=>k6.Acos,"Acosh",()=>k6.Acosh,"AdadeltaOptimizer",()=>k6.AdadeltaOptimizer,"AdagradOptimizer",()=>k6.AdagradOptimizer,"AdamOptimizer",()=>k6.AdamOptimizer,"AdamaxOptimizer",()=>k6.AdamaxOptimizer,"Add",()=>k6.Add,"AddN",()=>k6.AddN,"All",()=>k6.All,"Any",()=>k6.Any,"ArgMax",()=>k6.ArgMax,"ArgMin",()=>k6.ArgMin,"Asin",()=>k6.Asin,"Asinh",()=>k6.Asinh,"Atan",()=>k6.Atan,"Atan2",()=>k6.Atan2,"Atanh",()=>k6.Atanh,"AvgPool",()=>k6.AvgPool,"AvgPool3D",()=>k6.AvgPool3D,"AvgPool3DGrad",()=>k6.AvgPool3DGrad,"AvgPoolGrad",()=>k6.AvgPoolGrad,"BatchMatMul",()=>k6.BatchMatMul,"BatchToSpaceND",()=>k6.BatchToSpaceND,"Bincount",()=>k6.Bincount,"BitwiseAnd",()=>k6.BitwiseAnd,"BroadcastArgs",()=>k6.BroadcastArgs,"BroadcastTo",()=>k6.BroadcastTo,"Callback",0,u6,"CallbackList",0,sB,"Cast",()=>k6.Cast,"Ceil",()=>k6.Ceil,"ClipByValue",()=>k6.ClipByValue,"Complex",()=>k6.Complex,"ComplexAbs",()=>k6.ComplexAbs,"Concat",()=>k6.Concat,"Conv2D",()=>k6.Conv2D,"Conv2DBackpropFilter",()=>k6.Conv2DBackpropFilter,"Conv2DBackpropInput",()=>k6.Conv2DBackpropInput,"Conv3D",()=>k6.Conv3D,"Conv3DBackpropFilterV2",()=>k6.Conv3DBackpropFilterV2,"Conv3DBackpropInputV2",()=>k6.Conv3DBackpropInputV2,"Cos",()=>k6.Cos,"Cosh",()=>k6.Cosh,"CropAndResize",()=>k6.CropAndResize,"Cumprod",()=>k6.Cumprod,"Cumsum",()=>k6.Cumsum,"CustomCallback",0,sU,"DataStorage",()=>k6.DataStorage,"DenseBincount",()=>k6.DenseBincount,"DepthToSpace",()=>k6.DepthToSpace,"DepthwiseConv2dNative",()=>k6.DepthwiseConv2dNative,"DepthwiseConv2dNativeBackpropFilter",()=>k6.DepthwiseConv2dNativeBackpropFilter,"DepthwiseConv2dNativeBackpropInput",()=>k6.DepthwiseConv2dNativeBackpropInput,"Diag",()=>k6.Diag,"Dilation2D",()=>k6.Dilation2D,"Dilation2DBackpropFilter",()=>k6.Dilation2DBackpropFilter,"Dilation2DBackpropInput",()=>k6.Dilation2DBackpropInput,"Draw",()=>k6.Draw,"ENV",()=>k6.ENV,"EarlyStopping",0,u9,"Einsum",()=>k6.Einsum,"Elu",()=>k6.Elu,"EluGrad",()=>k6.EluGrad,"Environment",()=>k6.Environment,"Equal",()=>k6.Equal,"Erf",()=>k6.Erf,"Exp",()=>k6.Exp,"ExpandDims",()=>k6.ExpandDims,"Expm1",()=>k6.Expm1,"FFT",()=>k6.FFT,"Fill",()=>k6.Fill,"FlipLeftRight",()=>k6.FlipLeftRight,"Floor",()=>k6.Floor,"FloorDiv",()=>k6.FloorDiv,"FromPixels",()=>k6.FromPixels,"FusedBatchNorm",()=>k6.FusedBatchNorm,"FusedConv2D",()=>k6.FusedConv2D,"FusedDepthwiseConv2D",()=>k6.FusedDepthwiseConv2D,"GPGPUContext",0,x8,"GatherNd",()=>k6.GatherNd,"GatherV2",()=>k6.GatherV2,"GraphModel",()=>k8.GraphModel,"Greater",()=>k6.Greater,"GreaterEqual",()=>k6.GreaterEqual,"History",0,sG,"IFFT",()=>k6.IFFT,"Identity",()=>k6.Identity,"Imag",()=>k6.Imag,"InputSpec",0,sp,"IsFinite",()=>k6.IsFinite,"IsInf",()=>k6.IsInf,"IsNan",()=>k6.IsNan,"KernelBackend",()=>k6.KernelBackend,"LRN",()=>k6.LRN,"LRNGrad",()=>k6.LRNGrad,"LayerVariable",0,su,"LayersModel",0,iF,"LeakyRelu",()=>k6.LeakyRelu,"Less",()=>k6.Less,"LessEqual",()=>k6.LessEqual,"LinSpace",()=>k6.LinSpace,"Log",()=>k6.Log,"Log1p",()=>k6.Log1p,"LogSoftmax",()=>k6.LogSoftmax,"LogicalAnd",()=>k6.LogicalAnd,"LogicalNot",()=>k6.LogicalNot,"LogicalOr",()=>k6.LogicalOr,"LogicalXor",()=>k6.LogicalXor,"LowerBound",()=>k6.LowerBound,"MathBackendCPU",0,de,"MathBackendWebGL",0,bs,"MatrixBandPart",()=>k6.MatrixBandPart,"Max",()=>k6.Max,"MaxPool",()=>k6.MaxPool,"MaxPool3D",()=>k6.MaxPool3D,"MaxPool3DGrad",()=>k6.MaxPool3DGrad,"MaxPoolGrad",()=>k6.MaxPoolGrad,"MaxPoolWithArgmax",()=>k6.MaxPoolWithArgmax,"Maximum",()=>k6.Maximum,"Mean",()=>k6.Mean,"Min",()=>k6.Min,"Minimum",()=>k6.Minimum,"MirrorPad",()=>k6.MirrorPad,"Mod",()=>k6.Mod,"MomentumOptimizer",()=>k6.MomentumOptimizer,"Multinomial",()=>k6.Multinomial,"Multiply",()=>k6.Multiply,"Neg",()=>k6.Neg,"NonMaxSuppressionV3",()=>k6.NonMaxSuppressionV3,"NonMaxSuppressionV4",()=>k6.NonMaxSuppressionV4,"NonMaxSuppressionV5",()=>k6.NonMaxSuppressionV5,"NotEqual",()=>k6.NotEqual,"OP_SCOPE_SUFFIX",()=>k6.OP_SCOPE_SUFFIX,"OneHot",()=>k6.OneHot,"OnesLike",()=>k6.OnesLike,"Optimizer",()=>k6.Optimizer,"OptimizerConstructors",()=>k6.OptimizerConstructors,"Pack",()=>k6.Pack,"PadV2",()=>k6.PadV2,"Pool",()=>k6.Pool,"Pow",()=>k6.Pow,"Prelu",()=>k6.Prelu,"Prod",()=>k6.Prod,"RMSPropOptimizer",()=>k6.RMSPropOptimizer,"RNN",0,oF,"RaggedGather",()=>k6.RaggedGather,"RaggedRange",()=>k6.RaggedRange,"RaggedTensorToTensor",()=>k6.RaggedTensorToTensor,"Range",()=>k6.Range,"Rank",()=>k6.Rank,"Real",()=>k6.Real,"RealDiv",()=>k6.RealDiv,"Reciprocal",()=>k6.Reciprocal,"Reduction",()=>k6.Reduction,"Relu",()=>k6.Relu,"Relu6",()=>k6.Relu6,"Reshape",()=>k6.Reshape,"ResizeBilinear",()=>k6.ResizeBilinear,"ResizeBilinearGrad",()=>k6.ResizeBilinearGrad,"ResizeNearestNeighbor",()=>k6.ResizeNearestNeighbor,"ResizeNearestNeighborGrad",()=>k6.ResizeNearestNeighborGrad,"Reverse",()=>k6.Reverse,"RotateWithOffset",()=>k6.RotateWithOffset,"Round",()=>k6.Round,"Rsqrt",()=>k6.Rsqrt,"SGDOptimizer",()=>k6.SGDOptimizer,"ScatterNd",()=>k6.ScatterNd,"SearchSorted",()=>k6.SearchSorted,"Select",()=>k6.Select,"Selu",()=>k6.Selu,"Sequential",0,i_,"Sigmoid",()=>k6.Sigmoid,"Sign",()=>k6.Sign,"Sin",()=>k6.Sin,"Sinh",()=>k6.Sinh,"Slice",()=>k6.Slice,"Softmax",()=>k6.Softmax,"Softplus",()=>k6.Softplus,"SpaceToBatchND",()=>k6.SpaceToBatchND,"SparseFillEmptyRows",()=>k6.SparseFillEmptyRows,"SparseReshape",()=>k6.SparseReshape,"SparseSegmentMean",()=>k6.SparseSegmentMean,"SparseSegmentSum",()=>k6.SparseSegmentSum,"SparseToDense",()=>k6.SparseToDense,"SplitV",()=>k6.SplitV,"Sqrt",()=>k6.Sqrt,"Square",()=>k6.Square,"SquaredDifference",()=>k6.SquaredDifference,"StaticRegexReplace",()=>k6.StaticRegexReplace,"Step",()=>k6.Step,"StridedSlice",()=>k6.StridedSlice,"StringNGrams",()=>k6.StringNGrams,"StringSplit",()=>k6.StringSplit,"StringToHashBucketFast",()=>k6.StringToHashBucketFast,"Sub",()=>k6.Sub,"Sum",()=>k6.Sum,"SymbolicTensor",0,sc,"Tan",()=>k6.Tan,"Tanh",()=>k6.Tanh,"Tensor",()=>k6.Tensor,"TensorBuffer",()=>k6.TensorBuffer,"TensorScatterUpdate",()=>k6.TensorScatterUpdate,"Tile",()=>k6.Tile,"TopK",()=>k6.TopK,"Transform",()=>k6.Transform,"Transpose",()=>k6.Transpose,"Unique",()=>k6.Unique,"Unpack",()=>k6.Unpack,"UnsortedSegmentSum",()=>k6.UnsortedSegmentSum,"UpperBound",()=>k6.UpperBound,"Variable",()=>k6.Variable,"ZerosLike",()=>k6.ZerosLike,"_FusedMatMul",()=>k6._FusedMatMul,"abs",()=>k6.abs,"acos",()=>k6.acos,"acosh",()=>k6.acosh,"add",()=>k6.add,"addN",()=>k6.addN,"all",()=>k6.all,"any",()=>k6.any,"argMax",()=>k6.argMax,"argMin",()=>k6.argMin,"asin",()=>k6.asin,"asinh",()=>k6.asinh,"atan",()=>k6.atan,"atan2",()=>k6.atan2,"atanh",()=>k6.atanh,"avgPool",()=>k6.avgPool,"avgPool3d",()=>k6.avgPool3d,"backend",()=>k6.backend,"backend_util",()=>k6.backend_util,"basicLSTMCell",()=>k6.basicLSTMCell,"batchNorm",()=>k6.batchNorm,"batchNorm2d",()=>k6.batchNorm2d,"batchNorm3d",()=>k6.batchNorm3d,"batchNorm4d",()=>k6.batchNorm4d,"batchToSpaceND",()=>k6.batchToSpaceND,"bincount",()=>k6.bincount,"bitwiseAnd",()=>k6.bitwiseAnd,"booleanMaskAsync",()=>k6.booleanMaskAsync,"broadcastArgs",()=>k6.broadcastArgs,"broadcastTo",()=>k6.broadcastTo,"broadcast_util",()=>k6.broadcast_util,"browser",()=>k6.browser,"buffer",()=>k6.buffer,"callbacks",0,he,"cast",()=>k6.cast,"ceil",()=>k6.ceil,"clipByValue",()=>k6.clipByValue,"clone",()=>k6.clone,"complex",()=>k6.complex,"concat",()=>k6.concat,"concat1d",()=>k6.concat1d,"concat2d",()=>k6.concat2d,"concat3d",()=>k6.concat3d,"concat4d",()=>k6.concat4d,"constraints",()=>sO,"conv1d",()=>k6.conv1d,"conv2d",()=>k6.conv2d,"conv2dTranspose",()=>k6.conv2dTranspose,"conv3d",()=>k6.conv3d,"conv3dTranspose",()=>k6.conv3dTranspose,"copyRegisteredKernels",()=>k6.copyRegisteredKernels,"cos",()=>k6.cos,"cosh",()=>k6.cosh,"cosineWindow",()=>k6.cosineWindow,"cumprod",()=>k6.cumprod,"cumsum",()=>k6.cumsum,"customGrad",()=>k6.customGrad,"data",0,h4,"denseBincount",()=>k6.denseBincount,"deprecationWarn",()=>k6.deprecationWarn,"depthToSpace",()=>k6.depthToSpace,"depthwiseConv2d",()=>k6.depthwiseConv2d,"deregisterOp",()=>k7.deregisterOp,"device_util",()=>k6.device_util,"diag",()=>k6.diag,"dilation2d",()=>k6.dilation2d,"disableDeprecationWarnings",()=>k6.disableDeprecationWarnings,"dispose",()=>k6.dispose,"disposeVariables",()=>k6.disposeVariables,"div",()=>k6.div,"divNoNan",()=>k6.divNoNan,"dot",()=>k6.dot,"dropout",()=>k6.dropout,"einsum",()=>k6.einsum,"elu",()=>k6.elu,"enableDebugMode",()=>k6.enableDebugMode,"enableProdMode",()=>k6.enableProdMode,"enclosingPowerOfTwo",()=>k6.enclosingPowerOfTwo,"engine",()=>k6.engine,"ensureShape",()=>k6.ensureShape,"env",()=>k6.env,"equal",()=>k6.equal,"erf",()=>k6.erf,"euclideanNorm",()=>k6.euclideanNorm,"exp",()=>k6.exp,"expandDims",()=>k6.expandDims,"expm1",()=>k6.expm1,"eye",()=>k6.eye,"fft",()=>k6.fft,"fill",()=>k6.fill,"findBackend",()=>k6.findBackend,"findBackendFactory",()=>k6.findBackendFactory,"floor",()=>k6.floor,"floorDiv",()=>k6.floorDiv,"forceHalfFloat",0,bl,"fused",()=>k6.fused,"gather",()=>k6.gather,"gatherND",()=>k6.gatherND,"gather_util",()=>k6.gather_util,"getBackend",()=>k6.getBackend,"getGradient",()=>k6.getGradient,"getKernel",()=>k6.getKernel,"getKernelsForBackend",()=>k6.getKernelsForBackend,"gpgpu_util",()=>bi,"grad",()=>k6.grad,"grads",()=>k6.grads,"greater",()=>k6.greater,"greaterEqual",()=>k6.greaterEqual,"ifft",()=>k6.ifft,"imag",()=>k6.imag,"image",()=>k6.image,"inTopKAsync",()=>k6.inTopKAsync,"initializers",()=>sL,"input",0,iB,"io",()=>k6.io,"irfft",()=>k6.irfft,"isFinite",()=>k6.isFinite,"isInf",()=>k6.isInf,"isNaN",()=>k6.isNaN,"keep",()=>k6.keep,"kernel_impls",()=>k6.kernel_impls,"layers",()=>u2,"leakyRelu",()=>k6.leakyRelu,"less",()=>k6.less,"lessEqual",()=>k6.lessEqual,"linalg",()=>k6.linalg,"linspace",()=>k6.linspace,"loadGraphModel",()=>k8.loadGraphModel,"loadGraphModelSync",()=>k8.loadGraphModelSync,"loadLayersModel",0,iL,"localResponseNormalization",()=>k6.localResponseNormalization,"log",()=>k6.log,"log1p",()=>k6.log1p,"logSigmoid",()=>k6.logSigmoid,"logSoftmax",()=>k6.logSoftmax,"logSumExp",()=>k6.logSumExp,"logicalAnd",()=>k6.logicalAnd,"logicalNot",()=>k6.logicalNot,"logicalOr",()=>k6.logicalOr,"logicalXor",()=>k6.logicalXor,"losses",()=>k6.losses,"lowerBound",()=>k6.lowerBound,"matMul",()=>k6.matMul,"math",()=>k6.math,"max",()=>k6.max,"maxPool",()=>k6.maxPool,"maxPool3d",()=>k6.maxPool3d,"maxPoolWithArgmax",()=>k6.maxPoolWithArgmax,"maximum",()=>k6.maximum,"mean",()=>k6.mean,"memory",()=>k6.memory,"meshgrid",()=>k6.meshgrid,"metrics",()=>u3,"min",()=>k6.min,"minimum",()=>k6.minimum,"mirrorPad",()=>k6.mirrorPad,"mod",()=>k6.mod,"model",0,iM,"models",()=>u4,"moments",()=>k6.moments,"movingAverage",()=>k6.movingAverage,"mul",()=>k6.mul,"multiRNNCell",()=>k6.multiRNNCell,"multinomial",()=>k6.multinomial,"neg",()=>k6.neg,"nextFrame",()=>k6.nextFrame,"norm",()=>k6.norm,"notEqual",()=>k6.notEqual,"oneHot",()=>k6.oneHot,"ones",()=>k6.ones,"onesLike",()=>k6.onesLike,"op",()=>k6.op,"outerProduct",()=>k6.outerProduct,"pad",()=>k6.pad,"pad1d",()=>k6.pad1d,"pad2d",()=>k6.pad2d,"pad3d",()=>k6.pad3d,"pad4d",()=>k6.pad4d,"pool",()=>k6.pool,"pow",()=>k6.pow,"prelu",()=>k6.prelu,"print",()=>k6.print,"prod",()=>k6.prod,"profile",()=>k6.profile,"raggedGather",()=>k6.raggedGather,"raggedRange",()=>k6.raggedRange,"raggedTensorToTensor",()=>k6.raggedTensorToTensor,"rand",()=>k6.rand,"randomGamma",()=>k6.randomGamma,"randomNormal",()=>k6.randomNormal,"randomStandardNormal",()=>k6.randomStandardNormal,"randomUniform",()=>k6.randomUniform,"randomUniformInt",()=>k6.randomUniformInt,"range",()=>k6.range,"ready",()=>k6.ready,"real",()=>k6.real,"reciprocal",()=>k6.reciprocal,"registerBackend",()=>k6.registerBackend,"registerCallbackConstructor",0,iW,"registerGradient",()=>k6.registerGradient,"registerKernel",()=>k6.registerKernel,"registerOp",()=>k7.registerOp,"regularizers",()=>u5,"relu",()=>k6.relu,"relu6",()=>k6.relu6,"removeBackend",()=>k6.removeBackend,"reshape",()=>k6.reshape,"reverse",()=>k6.reverse,"reverse1d",()=>k6.reverse1d,"reverse2d",()=>k6.reverse2d,"reverse3d",()=>k6.reverse3d,"reverse4d",()=>k6.reverse4d,"rfft",()=>k6.rfft,"round",()=>k6.round,"rsqrt",()=>k6.rsqrt,"scalar",()=>k6.scalar,"scatterND",()=>k6.scatterND,"scatter_util",()=>k6.scatter_util,"searchSorted",()=>k6.searchSorted,"selu",()=>k6.selu,"separableConv2d",()=>k6.separableConv2d,"sequential",0,iP,"serialization",()=>k6.serialization,"setBackend",()=>k6.setBackend,"setPlatform",()=>k6.setPlatform,"setWebGLContext",0,gT,"setdiff1dAsync",()=>k6.setdiff1dAsync,"shared",()=>cs,"sigmoid",()=>k6.sigmoid,"sign",()=>k6.sign,"signal",()=>k6.signal,"sin",()=>k6.sin,"sinh",()=>k6.sinh,"slice",()=>k6.slice,"slice1d",()=>k6.slice1d,"slice2d",()=>k6.slice2d,"slice3d",()=>k6.slice3d,"slice4d",()=>k6.slice4d,"slice_util",()=>k6.slice_util,"softmax",()=>k6.softmax,"softplus",()=>k6.softplus,"spaceToBatchND",()=>k6.spaceToBatchND,"sparse",()=>k6.sparse,"sparseToDense",()=>k6.sparseToDense,"spectral",()=>k6.spectral,"split",()=>k6.split,"sqrt",()=>k6.sqrt,"square",()=>k6.square,"squaredDifference",()=>k6.squaredDifference,"squeeze",()=>k6.squeeze,"stack",()=>k6.stack,"step",()=>k6.step,"stridedSlice",()=>k6.stridedSlice,"string",()=>k6.string,"sub",()=>k6.sub,"sum",()=>k6.sum,"sumOutType",()=>k6.sumOutType,"tan",()=>k6.tan,"tanh",()=>k6.tanh,"tensor",()=>k6.tensor,"tensor1d",()=>k6.tensor1d,"tensor2d",()=>k6.tensor2d,"tensor3d",()=>k6.tensor3d,"tensor4d",()=>k6.tensor4d,"tensor5d",()=>k6.tensor5d,"tensor6d",()=>k6.tensor6d,"tensorScatterUpdate",()=>k6.tensorScatterUpdate,"tensor_util",()=>k6.tensor_util,"test_util",()=>k6.test_util,"tidy",()=>k6.tidy,"tile",()=>k6.tile,"time",()=>k6.time,"topk",()=>k6.topk,"train",()=>k6.train,"transpose",()=>k6.transpose,"truncatedNormal",()=>k6.truncatedNormal,"unique",()=>k6.unique,"unregisterGradient",()=>k6.unregisterGradient,"unregisterKernel",()=>k6.unregisterKernel,"unsortedSegmentSum",()=>k6.unsortedSegmentSum,"unstack",()=>k6.unstack,"upcastType",()=>k6.upcastType,"upperBound",()=>k6.upperBound,"util",()=>k6.util,"valueAndGrad",()=>k6.valueAndGrad,"valueAndGrads",()=>k6.valueAndGrads,"variable",()=>k6.variable,"variableGrads",()=>k6.variableGrads,"version",0,k5,"version_converter",0,k4,"version_core",()=>k6.version_core,"version_cpu",0,k2,"version_layers",0,id,"version_webgl",0,k3,"webgl",0,bu,"webgl_util",()=>bo,"where",()=>k6.where,"whereAsync",()=>k6.whereAsync,"zeros",()=>k6.zeros,"zerosLike",()=>k6.zerosLike],67271)}]);