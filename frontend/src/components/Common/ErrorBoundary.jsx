import React from "react";
export default class ErrorBoundary extends React.Component{constructor(p){super(p);this.state={hasError:false}}static getDerivedStateFromError(){return {hasError:true}}componentDidCatch(){}render(){if(this.state.hasError)return <div className="card text-danger">UI error occurred.</div>;return this.props.children;}}
